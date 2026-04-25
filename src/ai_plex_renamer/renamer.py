from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Iterable, Iterator, Optional

from .ai import NvidiaAIClassifier
from .debug import DebugLogger, debug_event
from .guessit_parser import guess_with_guessit
from .heuristics import clean_folder_title, guess_folder_episode_from_path, guess_from_filename
from .models import MediaGuess, RenamePlan
from .naming import build_plex_filename, build_plex_folder_name, is_video_file, resolve_collision
from .tmdb import TMDBClient


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

    if target.name == source.name:
        return RenamePlan(
            source=source,
            target=target,
            guess=guess,
            status="unchanged",
            message="Already matches the target Plex filename.",
        )

    return RenamePlan(source=source, target=target, guess=guess, status="planned")


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
    local_guesses = {file_path: _local_guess(file_path, library_hint) for file_path in files}
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
                    guesses[file_path] = _enrich_with_tmdb(ai_guess, tmdb_client)

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
    if source.parent != root:
        return source.parent
    if _folder_looks_like_show(source.parent.name, guess):
        return source.parent
    return source.parent / build_plex_folder_name(guess)


def _folder_looks_like_show(folder_name: str, guess: MediaGuess) -> bool:
    folder_title = _normalize_title_for_compare(clean_folder_title(folder_name))
    guess_title = _normalize_title_for_compare(guess.title)
    plex_folder = _normalize_title_for_compare(build_plex_folder_name(guess))
    return bool(folder_title and guess_title and folder_title in {guess_title, plex_folder})


def _normalize_title_for_compare(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.lower())


def apply_plan(plan: RenamePlan) -> RenamePlan:
    if plan.status != "planned" or plan.target is None:
        return plan
    try:
        plan.target.parent.mkdir(parents=True, exist_ok=True)
        plan.source.rename(plan.target)
    except OSError as exc:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="error",
            message=f"Rename failed: {exc}",
        )
    else:
        return RenamePlan(
            source=plan.source,
            target=plan.target,
            guess=plan.guess,
            status="renamed",
            message="Renamed successfully.",
        )


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _enrich_with_tmdb(guess: MediaGuess, tmdb_client: Optional[TMDBClient]) -> MediaGuess:
    if tmdb_client is None:
        return guess
    return tmdb_client.enrich(guess)
