from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import shutil
import time
from typing import Iterable, Iterator, Optional

from .ai import NvidiaAIClassifier
from .debug import DebugLogger, debug_event
from .guessit_parser import guess_with_guessit
from .heuristics import (
    clean_folder_title,
    clean_name_text,
    coerce_special_guess,
    guess_folder_episode_from_path,
    guess_from_filename,
    is_special_folder_name,
    is_unnumbered_special_path,
)
from .models import MediaGuess, RenamePlan
from .naming import build_plex_filename, build_plex_folder_name, is_video_file, resolve_collision
from .tmdb import TMDBClient


APPLY_RETRY_DELAYS = (0.2, 0.5)
SIDECAR_EXTENSIONS = {".ass", ".idx", ".smi", ".srt", ".ssa", ".sub", ".sup", ".vtt"}
SIDECAR_LANGUAGE_ALIASES = {
    "sc": "zh-Hans",
    "chs": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-hans": "zh-Hans",
    "tc": "zh-Hant",
    "cht": "zh-Hant",
    "zh-tw": "zh-Hant",
    "zh-hant": "zh-Hant",
}


def iter_media_files(root: Path, recursive: bool = True, include_hidden: bool = False) -> Iterator[Path]:
    if root.is_file():
        if is_video_file(root) and (include_hidden or not _is_hidden(root)):
            yield root
        return

    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if not include_hidden and _is_hidden(path):
            continue
        if is_video_file(path):
            yield path


def make_rename_plan(
    source: Path,
    root: Path,
    classifier: Optional[NvidiaAIClassifier],
    tmdb_client: Optional[TMDBClient] = None,
    library_hint: str = "auto",
    collision: str = "skip",
    organize_root_tv: bool = True,
    debug: Optional[DebugLogger] = None,
) -> RenamePlan:
    return _make_plans_for_group(
        [source],
        root=root,
        classifier=classifier,
        tmdb_client=tmdb_client,
        library_hint=library_hint,
        collision=collision,
        organize_root_tv=organize_root_tv,
        debug=debug,
    )[0]


def _plan_from_guess(
    source: Path,
    root: Path,
    guess: MediaGuess,
    collision: str,
    organize_root_tv: bool,
) -> RenamePlan:
    if not guess.is_usable():
        return RenamePlan(
            source=source,
            target=None,
            guess=guess,
            status="skipped",
            message=guess.reason or "Could not infer Plex metadata safely.",
        )

    if _should_skip_after_tmdb_failure(guess):
        return RenamePlan(
            source=source,
            target=None,
            guess=guess,
            status="skipped",
            message="TMDB lookup failed after retries; skipped low-confidence/default guess to avoid unsafe rename.",
        )

    try:
        new_name = build_plex_filename(guess, source.suffix)
        target_dir = _target_directory(source, root, guess, organize_root_tv)
        target = resolve_collision(target_dir / new_name, source, collision)
    except Exception as exc:
        return RenamePlan(
            source=source,
            target=None,
            guess=guess,
            status="skipped",
            message=str(exc),
        )

    if target == source:
        return RenamePlan(
            source=source,
            target=target,
            guess=guess,
            status="unchanged",
            message="Already matches the target Plex filename.",
        )

    return RenamePlan(source=source, target=target, guess=guess, status="planned")


def validate_apply_plans(plans: Iterable[RenamePlan]) -> list[str]:
    problems: list[str] = []
    planned_targets: dict[str, Path] = {}

    for plan in plans:
        if plan.status != "planned" or plan.target is None:
            continue

        source = _resolve_existing_source(plan.source)
        if source is None:
            problems.append(_rename_diagnostics("Source file was not found during apply preflight.", plan.source, plan.target))
            continue

        _validate_move_target(source, plan.target, planned_targets, problems)
        for sidecar_source, sidecar_target in _find_sidecar_moves(source, plan.target):
            _validate_move_target(sidecar_source, sidecar_target, planned_targets, problems)

    return problems


def apply_plans_by_group(
    plans: Iterable[RenamePlan],
    root: Path,
    failed_dir_name: str = ".failed",
    retry_delays: Iterable[float] = APPLY_RETRY_DELAYS,
) -> list[RenamePlan]:
    delays = tuple(retry_delays)
    groups: dict[Path, list[RenamePlan]] = {}
    for plan in plans:
        groups.setdefault(plan.source.parent, []).append(plan)

    results: list[RenamePlan] = []
    for group_plans in groups.values():
        results.extend(_apply_plan_group(group_plans, root, failed_dir_name, delays))
    return results


def _apply_plan_group(
    plans: list[RenamePlan],
    root: Path,
    failed_dir_name: str,
    retry_delays: Iterable[float],
) -> list[RenamePlan]:
    duplicate_plans = _duplicate_target_loser_plans(plans)
    duplicate_ids = {id(plan) for plan in duplicate_plans}
    outcomes: dict[int, RenamePlan] = {}

    for plan in duplicate_plans:
        outcomes[id(plan)] = _move_plan_to_failed(plan, root, failed_dir_name, retry_delays, reason="Duplicate target")

    kept_plans = [plan for plan in plans if id(plan) not in duplicate_ids]
    existing_conflict_ids: set[int] = set()
    for plan in kept_plans:
        if _has_existing_target_conflict(plan):
            outcomes[id(plan)] = _move_plan_to_failed(
                plan,
                root,
                failed_dir_name,
                retry_delays,
                reason="Target already exists",
            )
            existing_conflict_ids.add(id(plan))

    kept_plans = [plan for plan in kept_plans if id(plan) not in existing_conflict_ids]
    preflight_errors = validate_apply_plans(kept_plans)
    if preflight_errors:
        message = "Apply preflight failed for this folder. No files in this folder were renamed. " + " ".join(preflight_errors)
        for plan in kept_plans:
            if plan.status == "planned":
                outcomes[id(plan)] = RenamePlan(
                    source=plan.source,
                    target=plan.target,
                    guess=plan.guess,
                    status="error",
                    message=message,
                )

    stop_group = False
    for plan in kept_plans:
        if id(plan) in outcomes:
            continue
        if stop_group and plan.status == "planned":
            outcomes[id(plan)] = RenamePlan(
                source=plan.source,
                target=plan.target,
                guess=plan.guess,
                status="skipped",
                message="Not attempted because a previous apply in this folder failed.",
            )
            continue
        applied = apply_plan(plan, retry_delays=retry_delays)
        outcomes[id(plan)] = applied
        if applied.status == "error":
            stop_group = True

    return [outcomes.get(id(plan), plan) for plan in plans]


def _duplicate_target_loser_plans(plans: Iterable[RenamePlan]) -> list[RenamePlan]:
    seen: dict[str, RenamePlan] = {}
    losers: list[RenamePlan] = []
    for plan in plans:
        if plan.status != "planned" or plan.target is None:
            continue
        key = _path_key(plan.target)
        if key in seen:
            losers.append(plan)
            continue
        seen[key] = plan
    return losers


def _move_plan_to_failed(
    plan: RenamePlan,
    root: Path,
    failed_dir_name: str,
    retry_delays: Iterable[float],
    reason: str,
) -> RenamePlan:
    source = _resolve_existing_source(plan.source)
    if source is None:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=_rename_diagnostics("Duplicate target source was not found.", plan.source, plan.target or plan.source),
        )

    planned_targets: dict[str, Path] = {}
    try:
        failed_target = _resolve_available_target(
            _failed_target_for_source(source, root, failed_dir_name),
            source,
            planned_targets,
        )
        planned_targets[_path_key(failed_target)] = source
        moves = [(source, failed_target)]
        for sidecar_source in _find_sidecar_sources(source):
            sidecar_target = _resolve_available_target(
                failed_target.parent / sidecar_source.name,
                sidecar_source,
                planned_targets,
            )
            planned_targets[_path_key(sidecar_target)] = sidecar_source
            moves.append((sidecar_source, sidecar_target))
    except OSError as exc:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=f"Could not plan duplicate target move to failed folder: {exc}",
        )

    move_error = _move_paths_with_rollback(moves, retry_delays)
    if move_error is not None:
        return RenamePlan(
            source=plan.source,
            target=failed_target,
            guess=plan.guess,
            status="error",
            message=f"Could not move duplicate target source to failed folder: {move_error}",
        )

    moved_sidecars = len(moves) - 1
    message = f"{reason}; moved original file to failed folder: {failed_target}"
    if moved_sidecars:
        message = f"{message} Moved {moved_sidecars} sidecar file(s)."
    return RenamePlan(
        source=plan.source,
        target=failed_target,
        guess=plan.guess,
        status="failed",
        message=message,
    )


def _has_existing_target_conflict(plan: RenamePlan) -> bool:
    if plan.status != "planned" or plan.target is None:
        return False
    source = _resolve_existing_source(plan.source)
    if source is None:
        return False
    if _exists(plan.target) is True and plan.target != source:
        return True
    for sidecar_source, sidecar_target in _find_sidecar_moves(source, plan.target):
        if _exists(sidecar_target) is True and sidecar_target != sidecar_source:
            return True
    return False


def _failed_target_for_source(source: Path, root: Path, failed_dir_name: str) -> Path:
    failed_root = Path(failed_dir_name).expanduser()
    if not failed_root.is_absolute():
        failed_root = root / failed_root
    try:
        relative_source = source.relative_to(root)
    except ValueError:
        relative_source = Path(source.name)
    return failed_root / relative_source


def _resolve_available_target(base_target: Path, source: Path, planned_targets: dict[str, Path]) -> Path:
    for index in range(10000):
        if index == 0:
            candidate = base_target
        else:
            candidate = base_target.with_name(f"{base_target.stem} ({index}){base_target.suffix}")
        previous_source = planned_targets.get(_path_key(candidate))
        if previous_source is not None and previous_source != source:
            continue
        if candidate == source or _exists(candidate) is False:
            return candidate
    raise FileExistsError(f"No available failed target for {source}")


def _move_paths_with_rollback(moves: Iterable[tuple[Path, Path]], retry_delays: Iterable[float]) -> Optional[str]:
    moved_paths: list[tuple[Path, Path]] = []
    delays = tuple(retry_delays)
    for source, target in moves:
        if source == target:
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            rollback_error = _rollback_moved_paths(moved_paths)
            message = _rename_diagnostics("Could not create target folder.", source, target, exc)
            if rollback_error:
                message = f"{message} Rollback failed: {rollback_error}"
            return message
        if _exists(target) is True:
            rollback_error = _rollback_moved_paths(moved_paths)
            message = _rename_diagnostics("Target file already exists.", source, target)
            if rollback_error:
                message = f"{message} Rollback failed: {rollback_error}"
            return message
        move_error, _ = _move_path_with_retry(source, target, delays)
        if move_error is not None:
            rollback_error = _rollback_moved_paths(moved_paths)
            if rollback_error is None:
                return f"{move_error} Rolled back moved file(s)."
            return f"{move_error} Rollback failed: {rollback_error}"
        moved_paths.append((target, source))
    return None


def _local_guess(source: Path, library_hint: str) -> MediaGuess:
    if library_hint != "movie":
        folder_episode_guess = guess_folder_episode_from_path(source)
        if folder_episode_guess.is_usable():
            return folder_episode_guess

    heuristic_guess = guess_from_filename(source, library_hint)
    if heuristic_guess.is_usable() and (heuristic_guess.media_type == "tv" or library_hint == "tv"):
        return heuristic_guess

    guessit_guess = guess_with_guessit(source, library_hint)
    if guessit_guess.is_usable():
        return guessit_guess

    if heuristic_guess.is_usable():
        return heuristic_guess

    return guessit_guess if guessit_guess.reason else heuristic_guess


def _should_try_ai(source: Path, root: Path, guess: MediaGuess, library_hint: str) -> bool:
    if library_hint == "tv" and _is_usable_tv_guess(guess):
        return False
    if is_unnumbered_special_path(source):
        return False
    if not guess.is_usable():
        return True
    if library_hint == "tv" and guess.media_type != "tv":
        return True
    if guess.media_type == "movie" and guess.year is None and guess.confidence <= 0.6:
        return True
    if guess.media_type == "movie" and guess.year is None and _has_series_folder_context(source, root):
        return True
    if guess.confidence < 0.5:
        return True
    return False


def _has_series_folder_context(source: Path, root: Path) -> bool:
    try:
        source.parent.relative_to(root)
    except ValueError:
        return False
    return source.parent != root


def build_plans(
    files: Iterable[Path],
    root: Path,
    classifier: Optional[NvidiaAIClassifier],
    tmdb_client: Optional[TMDBClient],
    library_hint: str,
    collision: str,
    organize_root_tv: bool = True,
    debug: Optional[DebugLogger] = None,
) -> list[RenamePlan]:
    groups: dict[Path, list[Path]] = {}
    for file_path in files:
        groups.setdefault(file_path.parent, []).append(file_path)

    plans: list[RenamePlan] = []
    for group_files in groups.values():
        debug_event(
            debug,
            "planning group",
            {
                "parent": group_files[0].parent if group_files else None,
                "files": group_files,
            },
        )
        plans.extend(
            _make_plans_for_group(
                group_files,
                root=root,
                classifier=classifier,
                tmdb_client=tmdb_client,
                library_hint=library_hint,
                collision=collision,
                organize_root_tv=organize_root_tv,
                debug=debug,
            )
        )
    return plans


def _make_plans_for_group(
    files: list[Path],
    root: Path,
    classifier: Optional[NvidiaAIClassifier],
    tmdb_client: Optional[TMDBClient],
    library_hint: str,
    collision: str,
    organize_root_tv: bool,
    debug: Optional[DebugLogger] = None,
) -> list[RenamePlan]:
    raw_local_guesses = {
        file_path: coerce_special_guess(file_path, _local_guess(file_path, library_hint))
        for file_path in files
    }
    local_guesses = _with_tv_episode_defaults(files, root, raw_local_guesses, library_hint)
    debug_event(
        debug,
        "local guesses",
        {
            str(file_path): guess.to_dict()
            for file_path, guess in local_guesses.items()
        },
    )
    guesses = {
        file_path: _enrich_with_tmdb(guess, tmdb_client) if guess.is_usable() else guess
        for file_path, guess in local_guesses.items()
    }
    debug_event(
        debug,
        "guesses after tmdb",
        {
            str(file_path): guess.to_dict()
            for file_path, guess in guesses.items()
        },
    )
    needs_ai = [
        file_path
        for file_path, guess in guesses.items()
        if classifier is not None and _should_try_ai(file_path, root, guess, library_hint)
    ]
    debug_event(debug, "ai candidates", {"files": needs_ai})

    ai_error: Optional[Exception] = None
    if classifier is not None and needs_ai:
        try:
            ai_guesses = _classify_group(classifier, files, root, library_hint, guesses)
        except Exception as exc:
            ai_error = exc
        else:
            debug_event(
                debug,
                "ai guesses",
                {
                    str(file_path): guess.to_dict()
                    for file_path, guess in ai_guesses.items()
                },
            )
            for file_path in needs_ai:
                ai_guess = ai_guesses.get(file_path)
                if ai_guess and ai_guess.is_usable():
                    guesses[file_path] = _enrich_with_tmdb(coerce_special_guess(file_path, ai_guess), tmdb_client)
                elif ai_guess:
                    guesses[file_path] = coerce_special_guess(file_path, ai_guess)

    guesses = _with_tv_episode_defaults(files, root, guesses, library_hint)
    plans: list[RenamePlan] = []
    for file_path in files:
        guess = guesses[file_path]
        if ai_error is not None and file_path in needs_ai:
            if not guess.is_usable():
                plans.append(
                    RenamePlan(
                        source=file_path,
                        target=None,
                        guess=MediaGuess.unknown(str(ai_error)),
                        status="error",
                        message=f"NVIDIA AI failed and no local fallback was usable: {ai_error}",
                    )
                )
                continue
            guess = replace(guess, reason=f"{guess.reason} NVIDIA AI failed: {ai_error}".strip())
        plans.append(_plan_from_guess(file_path, root, guess, collision, organize_root_tv))
    debug_event(debug, "plans", [plan.to_dict() for plan in plans])
    return plans


def _with_tv_episode_defaults(
    files: list[Path],
    root: Path,
    guesses: dict[Path, MediaGuess],
    library_hint: str,
) -> dict[Path, MediaGuess]:
    if library_hint != "tv":
        return guesses

    candidates: dict[Path, str] = {}
    title_counts: dict[str, int] = {}
    for file_path in files:
        guess = guesses[file_path]
        if is_unnumbered_special_path(file_path) or _is_usable_tv_guess(guess):
            continue
        title = _single_tv_episode_title(file_path, root, guess)
        if not _is_clear_single_tv_title(title):
            continue
        key = _single_tv_episode_title_key(file_path, root, guess, title)
        if not key:
            continue
        candidates[file_path] = title
        title_counts[key] = title_counts.get(key, 0) + 1

    if not candidates:
        return guesses

    defaulted = dict(guesses)
    for file_path, title in candidates.items():
        key = _single_tv_episode_title_key(file_path, root, guesses[file_path], title)
        if title_counts[key] != 1:
            continue
        defaulted[file_path] = MediaGuess(
            media_type="tv",
            title=title,
            season=1,
            episode=1,
            confidence=0.55,
            reason=(
                "Defaulted to S01E01 because --type tv was provided and this file has a unique inferred title."
            ),
        )
    return defaulted


def _is_usable_tv_guess(guess: MediaGuess) -> bool:
    return guess.media_type == "tv" and guess.is_usable()


def _single_tv_episode_title(source: Path, root: Path, guess: MediaGuess) -> str:
    if guess.title:
        return clean_folder_title(guess.title)
    if source.parent != root:
        parent_title = clean_folder_title(source.parent.name)
        if parent_title:
            return parent_title
    return clean_folder_title(source.stem) or clean_name_text(source.stem)


def _single_tv_episode_title_key(source: Path, root: Path, guess: MediaGuess, title: str) -> str:
    compare_title = title
    if not guess.title and source.parent == root:
        compare_title = _strip_trailing_episode_marker(title) or title
    if not _is_clear_single_tv_title(compare_title):
        return ""
    return _normalize_title_for_compare(compare_title)


def _strip_trailing_episode_marker(title: str) -> str:
    return re.sub(
        r"(?ix)"
        r"(?:"
        r"\s*[-_. ]+\s*(?:episode|ep|e|\#)?\s*\d{1,3}"
        r"|"
        r"\s*第\s*[\d零〇一二三四五六七八九十百壱弐参兩两]{1,8}\s*(?:話|话|集|回|巻|卷)(?:\s*v\d{1,3})?"
        r")$",
        "",
        title,
    ).strip(" -_.")


def _is_clear_single_tv_title(title: str) -> bool:
    normalized = _normalize_title_for_compare(title)
    if not normalized or normalized.isdigit():
        return False
    if normalized in {"episode", "movie", "sample", "unknown", "video"}:
        return False
    return any(character.isalpha() for character in title)


def _should_skip_after_tmdb_failure(guess: MediaGuess) -> bool:
    reason = guess.reason or ""
    if "TMDB lookup failed:" not in reason:
        return False
    if reason.startswith("Defaulted to S01E01"):
        return True
    if guess.confidence <= 0.6:
        return True
    return False


def _classify_group(
    classifier: NvidiaAIClassifier,
    files: list[Path],
    root: Path,
    library_hint: str,
    local_guesses: dict[Path, MediaGuess],
) -> dict[Path, MediaGuess]:
    classify_many = getattr(classifier, "classify_many", None)
    if callable(classify_many):
        return classify_many(files, root, library_hint, local_guesses)

    return {
        file_path: classifier.classify(
            file_path,
            root=root,
            library_hint=library_hint,
            local_guess=local_guesses[file_path],
        )
        for file_path in files
        if _should_try_ai(file_path, root, local_guesses[file_path], library_hint)
    }


def _target_directory(source: Path, root: Path, guess: MediaGuess, organize_root_tv: bool) -> Path:
    if not organize_root_tv or guess.media_type != "tv":
        return source.parent
    if guess.season == 0:
        return _target_special_directory(source, root, guess)
    if source.parent != root:
        return source.parent
    if _folder_looks_like_show(source.parent.name, guess):
        return source.parent
    return source.parent / build_plex_folder_name(guess)


def _target_special_directory(source: Path, root: Path, guess: MediaGuess) -> Path:
    if is_special_folder_name(source.parent.name):
        if _normalize_title_for_compare(source.parent.name) in {"specials", "season00", "season0"}:
            return source.parent
        if source.parent != root:
            return source.parent.parent / "Specials"
    if source.parent == root:
        return root / build_plex_folder_name(guess) / "Specials"
    return source.parent


def _folder_looks_like_show(folder_name: str, guess: MediaGuess) -> bool:
    folder_title = _normalize_title_for_compare(clean_folder_title(folder_name))
    guess_title = _normalize_title_for_compare(guess.title)
    plex_folder = _normalize_title_for_compare(build_plex_folder_name(guess))
    return bool(folder_title and guess_title and folder_title in {guess_title, plex_folder})


def _normalize_title_for_compare(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.lower())


def apply_plan(plan: RenamePlan, retry_delays: Iterable[float] = APPLY_RETRY_DELAYS) -> RenamePlan:
    if plan.status != "planned" or plan.target is None:
        return plan

    source = _resolve_existing_source(plan.source)
    if source is None:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=_rename_diagnostics("Source file was not found at apply time.", plan.source, plan.target),
        )

    if plan.target.exists() and plan.target != source:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=_rename_diagnostics("Target file already exists at apply time.", source, plan.target),
        )

    delays = tuple(retry_delays)
    sidecar_moves = _find_sidecar_moves(source, plan.target)
    sidecar_conflict = _first_existing_target(sidecar_moves)
    if sidecar_conflict is not None:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=_rename_diagnostics(
                "Target sidecar file already exists at apply time.",
                sidecar_conflict[0],
                sidecar_conflict[1],
            ),
        )
    sidecar_duplicate = _first_duplicate_target(sidecar_moves)
    if sidecar_duplicate is not None:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=_rename_diagnostics(
                "Multiple sidecar files would use the same target filename.",
                sidecar_duplicate[0],
                sidecar_duplicate[2],
            ),
        )

    try:
        plan.target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=_rename_diagnostics("Could not create target folder.", source, plan.target, exc),
        )

    moved_paths: list[tuple[Path, Path]] = []
    move_error, used_fallback = _move_path_with_retry(source, plan.target, delays)
    if move_error is not None:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=move_error,
        )
    moved_paths.append((plan.target, source))

    moved_sidecars = 0
    for sidecar_source, sidecar_target in sidecar_moves:
        sidecar_error, _ = _move_path_with_retry(sidecar_source, sidecar_target, delays)
        if sidecar_error is not None:
            rollback_error = _rollback_moved_paths(moved_paths)
            message = f"Sidecar move failed: {sidecar_error}"
            if rollback_error is None:
                message = f"{message} Rolled back moved file(s)."
            else:
                message = f"{message} Rollback failed: {rollback_error}"
            return RenamePlan(
                source=plan.source,
                target=plan.target,
                guess=plan.guess,
                status="error",
                message=message,
            )
        moved_paths.append((sidecar_target, sidecar_source))
        moved_sidecars += 1

    message = "Renamed successfully via copy fallback." if used_fallback else "Renamed successfully."
    if moved_sidecars:
        message = f"{message} Moved {moved_sidecars} sidecar file(s)."
    return RenamePlan(
        source=plan.source,
        target=plan.target,
        guess=plan.guess,
        status="renamed",
        message=message,
    )


def _rollback_moved_paths(moved_paths: Iterable[tuple[Path, Path]]) -> Optional[str]:
    errors: list[str] = []
    for current, original in reversed(tuple(moved_paths)):
        if _exists(current) is not True:
            errors.append(f"rollback_source_missing={current}")
            continue
        if _exists(original) is True:
            errors.append(f"rollback_target_exists={original}")
            continue
        error, _ = _move_path_with_retry(current, original, ())
        if error is not None:
            errors.append(error)
    return "; ".join(errors) if errors else None


def _resolve_existing_source(source: Path) -> Optional[Path]:
    if _exists(source):
        return source
    parent = source.parent
    if not _exists(parent):
        return None
    try:
        for candidate in parent.iterdir():
            if candidate.name.casefold() == source.name.casefold() and candidate.is_file():
                return candidate
    except OSError:
        return None
    return None


def _move_path_with_retry(source: Path, target: Path, retry_delays: Iterable[float]) -> tuple[Optional[str], bool]:
    delays = tuple(retry_delays)
    for attempt in range(len(delays) + 1):
        try:
            source.rename(target)
        except FileNotFoundError as exc:
            refreshed_source = _resolve_existing_source(source)
            if refreshed_source is not None:
                source = refreshed_source
            if attempt < len(delays):
                time.sleep(delays[attempt])
                continue
            fallback_error = _copy_move_fallback(source, target)
            if fallback_error is None:
                return None, True
            return _rename_diagnostics(
                "Rename failed because Windows could not find the source path.",
                source,
                target,
                exc,
            ), False
        except OSError as exc:
            return _rename_diagnostics("Rename failed.", source, target, exc), False
        else:
            return None, False
    return _rename_diagnostics("Rename failed for an unknown reason.", source, target), False


def _copy_move_fallback(source: Path, target: Path) -> Optional[str]:
    if _exists(source) is not True or _exists(target) is True:
        return _rename_diagnostics("Copy fallback was not usable.", source, target)
    try:
        shutil.move(str(source), str(target))
    except OSError as exc:
        return _rename_diagnostics("Rename failed and copy fallback failed.", source, target, exc)
    return None


def _find_sidecar_moves(source: Path, target: Path) -> list[tuple[Path, Path]]:
    moves: list[tuple[Path, Path]] = []
    for candidate in _find_sidecar_sources(source):
        suffix = candidate.name[len(source.stem):]
        moves.append((candidate, target.with_name(f"{target.stem}{_normalize_sidecar_suffix(suffix)}")))
    return moves


def _find_sidecar_sources(source: Path) -> list[Path]:
    parent = source.parent
    if not _exists(parent):
        return []
    prefix = f"{source.stem}."
    try:
        candidates = sorted(parent.iterdir(), key=lambda path: path.name.casefold())
    except OSError:
        return []
    sidecars: list[Path] = []
    for candidate in candidates:
        if candidate == source or not candidate.is_file():
            continue
        if not _is_sidecar_for_source(candidate, prefix):
            continue
        sidecars.append(candidate)
    return sidecars


def _is_sidecar_for_source(candidate: Path, prefix: str) -> bool:
    if candidate.suffix.lower() not in SIDECAR_EXTENSIONS:
        return False
    return candidate.name.casefold().startswith(prefix.casefold())


def _normalize_sidecar_suffix(suffix: str) -> str:
    parts = suffix.split(".")
    if len(parts) < 3 or parts[0] != "":
        return suffix

    language = parts[1].lower()
    normalized_language = SIDECAR_LANGUAGE_ALIASES.get(language)
    if normalized_language is None:
        return suffix
    return "." + ".".join([normalized_language, *parts[2:]])


def _first_existing_target(moves: Iterable[tuple[Path, Path]]) -> Optional[tuple[Path, Path]]:
    for source, target in moves:
        if _exists(target) is True and target != source:
            return source, target
    return None


def _first_duplicate_target(moves: Iterable[tuple[Path, Path]]) -> Optional[tuple[Path, Path, Path]]:
    seen: dict[str, Path] = {}
    for source, target in moves:
        key = _path_key(target)
        previous_source = seen.get(key)
        if previous_source is not None:
            return previous_source, source, target
        seen[key] = source
    return None


def _validate_move_target(
    source: Path,
    target: Path,
    planned_targets: dict[str, Path],
    problems: list[str],
) -> None:
    key = _path_key(target)
    previous_source = planned_targets.get(key)
    if previous_source is not None:
        problems.append(f"Duplicate target planned: {target} from {previous_source} and {source}")
        return
    planned_targets[key] = source

    if _exists(target.parent) is True and not target.parent.is_dir():
        problems.append(_rename_diagnostics("Target parent exists but is not a folder during apply preflight.", source, target))

    if _exists(target) is True and target != source:
        problems.append(_rename_diagnostics("Target file already exists during apply preflight.", source, target))


def _rename_diagnostics(message: str, source: Path, target: Path, exc: Optional[BaseException] = None) -> str:
    parts = [message]
    if exc is not None:
        parts.append(str(exc))
    parts.extend(
        [
            f"source_exists={_exists(source)}",
            f"target_parent_exists={_exists(target.parent)}",
            f"target_exists={_exists(target)}",
        ]
    )
    return " ".join(parts)


def _path_key(path: Path) -> str:
    return str(path).casefold()


def _exists(path: Path) -> bool | str:
    try:
        return path.exists()
    except OSError as exc:
        return f"error:{exc}"


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _enrich_with_tmdb(guess: MediaGuess, tmdb_client: Optional[TMDBClient]) -> MediaGuess:
    if tmdb_client is None:
        return guess
    return tmdb_client.enrich(guess)
