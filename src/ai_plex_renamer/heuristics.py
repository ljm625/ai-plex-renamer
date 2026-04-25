from __future__ import annotations

from dataclasses import replace
import re
import unicodedata
from pathlib import Path
from typing import Optional

from .models import MediaGuess


CURRENT_MAX_YEAR = 2035
TV_PATTERNS = [
    re.compile(
        r"(?ix)"
        r"^(?P<title>.*?)"
        r"(?:^|[\s._\-\[\(])"
        r"s(?P<season>\d{1,2})[\s._\-]*e(?P<episode>\d{1,3})"
        r"(?:[\s._\-]*(?:e|-)?(?P<episode_end>\d{1,3}))?"
    ),
    re.compile(
        r"(?ix)"
        r"^(?P<title>.*?)"
        r"(?:^|[\s._\-\[\(])"
        r"(?P<season>\d{1,2})x(?P<episode>\d{1,3})"
        r"(?:[\s._\-]*(?:x|-)?(?P<episode_end>\d{1,3}))?"
    ),
    re.compile(
        r"(?ix)"
        r"^(?P<title>.*?)"
        r"(?:season|series|第)\s*(?P<season>\d{1,2})\s*(?:季|[\s._\-]*)"
        r"(?:episode|ep|e|第)\s*(?P<episode>\d{1,3})\s*(?:集)?"
    ),
]
FOLDER_EPISODE_PATTERNS = [
    (re.compile(r"(?i)(?:第\s*)?0*1\s*(?:話|话|集|回|episode|ep\b)"), 1, None),
    (re.compile(r"(?i)(?:第\s*)?0*2\s*(?:話|话|集|回|episode|ep\b)"), 2, None),
    (re.compile(r"(?i)(?:第\s*)?0*3\s*(?:話|话|集|回|episode|ep\b)"), 3, None),
    (re.compile(r"(?i)(?:前編|前篇|上巻|上集|上篇|part[\s._-]*1|pt[\s._-]*1)"), 1, "前編"),
    (re.compile(r"(?i)(?:中編|中篇|中巻|中集|part[\s._-]*2|pt[\s._-]*2)"), 2, "中編"),
    (re.compile(r"(?i)(?:part[\s._-]*3|pt[\s._-]*3)"), 3, None),
    (re.compile(r"(?i)(?:後編|后編|後篇|后篇|下巻|下集|下篇|part[\s._-]*2|pt[\s._-]*2)"), 2, "後編"),
]
HASH_EPISODE_PATTERN = re.compile(r"(?i)(?:[#＃]\s*|(?:episode|ep)\s*\.?\s*)(?P<episode>\d{1,3})")
BRACKET_EPISODE_PATTERN = re.compile(r"(?i)[\[【(]\s*(?P<episode>\d{1,3})\s*[\]】)]")
BRACKET_TOKEN_PATTERN = re.compile(r"[\[【(]\s*(?P<token>[^\]\】\)]{1,120}?)\s*[\]】)]")
EPISODE_TOKEN_PATTERN = re.compile(r"^\d{1,3}$")
EPISODE_RANGE_TOKEN_PATTERN = re.compile(r"^\d{1,3}\s*-\s*\d{1,3}$")
NUMBERED_EPISODE_PATTERN = re.compile(r"(?i)(?:第\s*)?(?P<episode>\d{1,3})\s*(?:話|话|集|回)")
EPISODE_AFTER_TITLE_TEMPLATE = r"(?i)^{title}(?:\s*[-_.]\s*|\s+)(?:episode|ep|e)?\s*(?P<episode>\d{{1,3}})(?=$|[\s._\-\(\[\{{])"
FOLDER_TITLE_PREFIX_PATTERN = re.compile(r"(?i)^(?:ova|oav|oad|tv|series|movie)\s+")
SPECIAL_FOLDER_NAMES = {"s0", "s00", "sp", "sps", "special", "specials", "season0", "season00"}
SPECIAL_MARKER_PATTERN = re.compile(
    r"(?i)^(?P<label>cm|sp|special|ova|oad|ncop|nced|op|ed|pv)(?P<number>\d{1,3})$"
)
UNNUMBERED_SPECIAL_MARKERS = {"commentary", "menu"}
LANGUAGE_TAGS = {
    "big5",
    "chs",
    "cht",
    "cn",
    "eng",
    "gb",
    "gb2312",
    "jpn",
    "jp",
    "sc",
    "tc",
    "zh",
    "zh-cn",
    "zh-hans",
    "zh-hant",
    "zh-tw",
}
YEAR_PATTERN = re.compile(r"(?<!\d)(?P<year>19\d{2}|20\d{2})(?!\d)")
RELEASE_TOKEN_PATTERN = re.compile(
    r"(?ix)"
    r"\b("
    r"2160p|1080p|720p|480p|4k|8k|uhd|hdr|dv|dolby|vision|"
    r"web[-_. ]?dl|webrip|web|blu[-_. ]?ray|bdrip|br[-_. ]?rip|remux|"
    r"hdtv|hdrip|dvdrip|xvid|x264|x265|h[._ ]?264|h[._ ]?265|hevc|avc|"
    r"aac|ac3|eac3|dts|truehd|atmos|ddp?5[._ ]?1|ddp?7[._ ]?1|"
    r"proper|repack|extended|unrated|internal|multi|subbed|dubbed|"
    r"nf|netflix|amzn|amazon|dsnp|disney|hmax|hulu|atvp|itunes"
    r")\b.*$"
)


def guess_from_filename(path: Path, library_hint: str = "auto") -> MediaGuess:
    stem = path.stem
    parent = path.parent.name if path.parent else ""
    hint = library_hint.lower()

    if hint != "movie":
        special_guess = guess_special_from_path(path)
        if special_guess.is_usable() or is_unnumbered_special_path(path):
            return special_guess

    folder_episode_guess = guess_folder_episode_from_path(path)
    if folder_episode_guess.is_usable():
        return folder_episode_guess

    tv_guess = _guess_tv(stem, fallback_title=clean_folder_title(parent))
    if tv_guess.is_usable():
        return tv_guess

    if hint == "tv":
        return MediaGuess.unknown("Could not find a season/episode pattern.")

    movie_guess = _guess_movie(stem, parent)
    if movie_guess.is_usable():
        return movie_guess

    if hint == "movie":
        title = clean_name_text(stem)
        if title:
            return MediaGuess(
                media_type="movie",
                title=title,
                confidence=0.35,
                reason="Treated as a movie because --type movie was provided.",
            )

    return MediaGuess.unknown("No common TV or movie filename pattern matched.")


def guess_folder_episode_from_path(path: Path) -> MediaGuess:
    return _guess_folder_episode(path.stem, path.parent.name if path.parent else "")


def guess_special_from_path(path: Path) -> MediaGuess:
    marker = _special_marker_from_path(path)
    if marker is None:
        return MediaGuess.unknown("No special marker matched.")

    label, episode = marker
    title = _title_for_special_path(path)
    if not title:
        return MediaGuess.unknown("Special marker found but no show title was clear.")
    if episode is None:
        return MediaGuess.unknown(
            f"Special/extra marker {label} has no episode number; skipped to avoid unsafe Plex special numbering."
        )

    return MediaGuess(
        media_type="tv",
        title=title,
        season=0,
        episode=episode,
        episode_title=label,
        confidence=0.76,
        reason="Matched a Plex special marker; Plex specials use Season 00.",
    )


def coerce_special_guess(path: Path, guess: MediaGuess) -> MediaGuess:
    marker = _special_marker_from_path(path)
    if marker is None:
        return guess

    label, episode = marker
    if episode is None:
        return MediaGuess.unknown(
            f"Special/extra marker {label} has no episode number; skipped to avoid unsafe Plex special numbering."
        )

    title = guess.title or _title_for_special_path(path)
    if not title:
        return guess
    if guess.media_type == "tv" and guess.season == 0 and guess.episode == episode:
        return guess
    return replace(
        guess,
        media_type="tv",
        title=title,
        season=0,
        episode=episode,
        episode_end=None,
        episode_title=guess.episode_title or label,
        confidence=max(guess.confidence, 0.76),
        reason=_join_reason(guess.reason, "Forced to Plex Season 00 because the path is a special."),
    )


def is_unnumbered_special_path(path: Path) -> bool:
    marker = _special_marker_from_path(path)
    return marker is not None and marker[1] is None


def is_special_folder_name(value: str) -> bool:
    return _normalize_title_for_compare(clean_name_text(value)) in SPECIAL_FOLDER_NAMES


def clean_name_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    value = value.replace("_", " ").replace(".", " ")
    value = re.sub(r"[\[\]\(\)\{\}]", " ", value)
    value = RELEASE_TOKEN_PATTERN.sub("", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" -.")


def is_language_tag(value: str) -> bool:
    normalized = clean_name_text(value).lower()
    normalized = normalized.replace(" ", "-")
    if normalized in LANGUAGE_TAGS:
        return True
    compact = normalized.replace("-", "").replace("&", "")
    return compact in LANGUAGE_TAGS or compact in {"chscht", "chtchs", "sctc", "tcsc"}


def _guess_tv(stem: str, fallback_title: str = "") -> MediaGuess:
    for pattern in TV_PATTERNS:
        match = pattern.search(stem)
        if not match:
            continue
        title = clean_name_text(match.group("title"))
        if not title:
            title = fallback_title or "Unknown Show"
        episode_title = _episode_title_after_match(stem, match.end())
        return MediaGuess(
            media_type="tv",
            title=title,
            season=_to_int(match.group("season")),
            episode=_to_int(match.group("episode")),
            episode_end=_to_int(match.groupdict().get("episode_end")),
            episode_title=episode_title,
            confidence=0.65,
            reason="Matched a common TV season/episode pattern.",
        )
    return MediaGuess.unknown("No TV pattern matched.")


def _guess_folder_episode(stem: str, parent: str) -> MediaGuess:
    title = clean_folder_title(parent)
    if not title or title in {".", ".."}:
        return MediaGuess.unknown("No parent folder title found.")

    cleaned_stem = clean_name_text(stem)
    if not cleaned_stem:
        return MediaGuess.unknown("No filename text found.")

    episode = _episode_number_from_marker(stem)
    if episode is None:
        episode = _episode_number_from_marker(cleaned_stem)
    if episode is None:
        episode = _episode_number_after_folder_title(cleaned_stem, title)
    episode_title = None
    if episode is None:
        for pattern, candidate_episode, candidate_title in FOLDER_EPISODE_PATTERNS:
            match = pattern.search(cleaned_stem)
            if not match:
                continue
            episode = candidate_episode
            episode_title = candidate_title or match.group(0)
            break

    if episode is None:
        return MediaGuess.unknown("No folder episode marker matched.")

    bracket_title = _title_from_filename_brackets(stem, title, episode)
    if bracket_title:
        title = bracket_title

    return MediaGuess(
        media_type="tv",
        title=title,
        season=1,
        episode=episode,
        episode_title=episode_title,
        confidence=0.7,
        reason="Inferred episode from parent folder and filename part marker.",
    )


def clean_folder_title(value: str) -> str:
    title = _clean_noisy_series_title(value) or clean_name_text(value)
    title = FOLDER_TITLE_PREFIX_PATTERN.sub("", title).strip()
    return title


def _episode_number_from_marker(value: str) -> Optional[int]:
    for pattern in (HASH_EPISODE_PATTERN, BRACKET_EPISODE_PATTERN, NUMBERED_EPISODE_PATTERN):
        match = pattern.search(value)
        if not match:
            continue
        episode = _to_int(match.group("episode"))
        if episode is not None:
            return episode
    return None


def _episode_number_after_folder_title(value: str, title: str) -> Optional[int]:
    if not value or not title:
        return None
    pattern = re.compile(EPISODE_AFTER_TITLE_TEMPLATE.format(title=re.escape(title)))
    match = pattern.search(value)
    if not match:
        return None
    return _to_int(match.group("episode"))


def _special_marker_from_path(path: Path) -> Optional[tuple[str, Optional[int]]]:
    tokens = [clean_name_text(match.group("token")) for match in BRACKET_TOKEN_PATTERN.finditer(path.stem)]
    for token in tokens:
        marker = _special_marker_from_token(token)
        if marker is not None:
            return marker
    if is_special_folder_name(path.parent.name if path.parent else ""):
        for token in tokens:
            if _normalize_title_for_compare(token) in UNNUMBERED_SPECIAL_MARKERS:
                return (token, None)
    return None


def _special_marker_from_token(token: str) -> Optional[tuple[str, Optional[int]]]:
    normalized = clean_name_text(token)
    normalized_key = _normalize_title_for_compare(normalized)
    if normalized_key in UNNUMBERED_SPECIAL_MARKERS:
        return (normalized, None)
    match = SPECIAL_MARKER_PATTERN.match(normalized_key)
    if not match:
        return None
    label = match.group("label").upper()
    number = _to_int(match.group("number"))
    return (f"{label}{number:02d}" if number is not None else label, number)


def _title_for_special_path(path: Path) -> str:
    if is_special_folder_name(path.parent.name if path.parent else "") and path.parent.parent != path.parent:
        title = clean_folder_title(path.parent.parent.name)
        if title:
            return title

    marker = _special_marker_from_path(path)
    if marker is not None and marker[1] is not None:
        bracket_title = _title_from_filename_brackets(path.stem, clean_folder_title(path.parent.name), marker[1])
        if bracket_title:
            return bracket_title

    return clean_folder_title(path.parent.name if path.parent else "")


def _title_from_filename_brackets(stem: str, parent_title: str, episode: int) -> str:
    tokens = [clean_name_text(match.group("token")) for match in BRACKET_TOKEN_PATTERN.finditer(stem)]
    episode_index = _episode_token_index(tokens, episode)
    if episode_index is None or episode_index == 0:
        return ""

    parent_normalized = _normalize_title_for_compare(parent_title)
    matches: list[str] = []
    for token in tokens[:episode_index]:
        if _is_bad_bracket_title_token(token):
            continue
        token_normalized = _normalize_title_for_compare(token)
        if not token_normalized:
            continue
        if parent_normalized and (token_normalized in parent_normalized or parent_normalized in token_normalized):
            matches.append(token)

    return matches[-1] if matches else ""


def _episode_token_index(tokens: list[str], episode: int) -> Optional[int]:
    for index, token in enumerate(tokens):
        if not EPISODE_TOKEN_PATTERN.match(token):
            continue
        if _to_int(token) == episode:
            return index
    return None


def _is_bad_bracket_title_token(value: str) -> bool:
    if not value:
        return True
    if is_language_tag(value):
        return True
    if EPISODE_TOKEN_PATTERN.match(value) or EPISODE_RANGE_TOKEN_PATTERN.match(value):
        return True
    if _special_marker_from_token(value) is not None:
        return True
    return not clean_name_text(value)


def _clean_noisy_series_title(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = re.sub(r"^(?:\s*[\[【(][^\]\】\)]{1,120}[\]】)]\s*)+", "", text)

    def replace_bracket(match: re.Match[str]) -> str:
        token = clean_name_text(match.group("token"))
        return " " if _is_noisy_bracket_token(token) else f" {token} "

    return clean_name_text(BRACKET_TOKEN_PATTERN.sub(replace_bracket, text))


def _is_noisy_bracket_token(value: str) -> bool:
    if not value:
        return True
    normalized = _normalize_title_for_compare(value)
    if is_language_tag(value):
        return True
    if EPISODE_TOKEN_PATTERN.match(value) or EPISODE_RANGE_TOKEN_PATTERN.match(value):
        return True
    if _special_marker_from_token(value) is not None:
        return True
    if normalized in {"unc", "end", "premium"}:
        return True
    if re.search(r"(?i)(?:\d{3,4}p|webdl|webrip|bluray|bdrip|ma\d+p|x26[45]|h26[45]|hevc|avc|flac|aac)", value):
        return True
    return False


def _normalize_title_for_compare(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.lower())


def _join_reason(*reasons: str) -> str:
    return " ".join(reason.strip() for reason in reasons if reason and reason.strip())


def _guess_movie(stem: str, parent: str) -> MediaGuess:
    year_match = _last_year_match(stem) or _last_year_match(parent)
    if not year_match:
        return MediaGuess.unknown("No movie year found.")

    year = _to_int(year_match.group("year"))
    if year is None or year < 1900 or year > CURRENT_MAX_YEAR:
        return MediaGuess.unknown("Movie year is outside the expected range.")

    title_source = stem[: year_match.start()] if year_match.string == stem else parent[: year_match.start()]
    title = clean_name_text(title_source)
    if not title:
        title = clean_name_text(stem)
        title = YEAR_PATTERN.sub("", title).strip()

    return MediaGuess(
        media_type="movie",
        title=title,
        year=year,
        confidence=0.55,
        reason="Matched a movie year pattern.",
    )


def _episode_title_after_match(stem: str, start: int) -> Optional[str]:
    tail = stem[start:]
    tail = RELEASE_TOKEN_PATTERN.sub("", tail)
    tail = clean_name_text(tail)
    if is_language_tag(tail):
        return None
    return tail or None


def _last_year_match(value: str) -> Optional[re.Match[str]]:
    matches = list(YEAR_PATTERN.finditer(value))
    return matches[-1] if matches else None


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
