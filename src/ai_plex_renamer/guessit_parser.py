from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .heuristics import clean_name_text, is_language_tag
from .models import MediaGuess


GuessItParser = Callable[..., Mapping[str, Any]]


def guess_with_guessit(
    path: Path,
    library_hint: str = "auto",
    parser: Optional[GuessItParser] = None,
) -> MediaGuess:
    try:
        raw_guess = _run_guessit(path, library_hint, parser)
    except ImportError:
        return MediaGuess.unknown("GuessIt is not installed.")
    except Exception as exc:
        return MediaGuess.unknown(f"GuessIt failed: {exc}")

    return media_guess_from_guessit(raw_guess, library_hint)


def media_guess_from_guessit(raw_guess: Mapping[str, Any], library_hint: str = "auto") -> MediaGuess:
    hint = library_hint.lower()
    raw_type = _first_string(raw_guess.get("type")).lower()

    if raw_type == "episode" or hint == "tv":
        return _episode_from_guessit(raw_guess)
    if raw_type == "movie" or hint == "movie":
        return _movie_from_guessit(raw_guess)

    episode_guess = _episode_from_guessit(raw_guess)
    if episode_guess.is_usable():
        return episode_guess

    movie_guess = _movie_from_guessit(raw_guess)
    if movie_guess.is_usable():
        return movie_guess

    return MediaGuess.unknown("GuessIt did not return usable Plex metadata.")


def _run_guessit(
    path: Path,
    library_hint: str,
    parser: Optional[GuessItParser],
) -> Mapping[str, Any]:
    if parser is None:
        from guessit import guessit as parser

    options = _guessit_options(library_hint)
    return parser(path.name, options) if options else parser(path.name)


def _guessit_options(library_hint: str) -> Optional[dict[str, Any]]:
    hint = library_hint.lower()
    if hint == "tv":
        return {"type": "episode"}
    if hint == "movie":
        return {"type": "movie"}
    return None


def _episode_from_guessit(raw_guess: Mapping[str, Any]) -> MediaGuess:
    series_value = raw_guess.get("series")
    title_value = raw_guess.get("title")
    title = _clean_title(series_value or title_value)
    if is_language_tag(title):
        title = ""
    season = _first_int(raw_guess.get("season") or raw_guess.get("seasonNumber"))
    episode_numbers = _int_list(raw_guess.get("episode") or raw_guess.get("episodeNumber"))
    episode = episode_numbers[0] if episode_numbers else None
    episode_end = episode_numbers[-1] if len(episode_numbers) > 1 else None
    episode_title = _clean_optional(raw_guess.get("episode_title") or raw_guess.get("episodeTitle"))
    if not episode_title and series_value and title_value:
        possible_episode_title = _clean_title(title_value)
        if possible_episode_title and possible_episode_title != title and not is_language_tag(possible_episode_title):
            episode_title = possible_episode_title

    if not title or season is None or episode is None:
        return MediaGuess.unknown("GuessIt did not find title, season, and episode.")

    return MediaGuess(
        media_type="tv",
        title=title,
        season=season,
        episode=episode,
        episode_end=episode_end,
        episode_title=episode_title,
        confidence=0.85,
        reason="Parsed by GuessIt.",
    )


def _movie_from_guessit(raw_guess: Mapping[str, Any]) -> MediaGuess:
    title = _clean_title(raw_guess.get("title"))
    if is_language_tag(title):
        return MediaGuess.unknown("GuessIt title is only a language/subtitle tag.")
    year = _first_int(raw_guess.get("year"))

    if not title:
        return MediaGuess.unknown("GuessIt did not find a movie title.")

    return MediaGuess(
        media_type="movie",
        title=title,
        year=year,
        confidence=0.8 if year is not None else 0.55,
        reason="Parsed by GuessIt.",
    )


def _clean_title(value: Any) -> str:
    text = _first_string(value)
    return clean_name_text(text)


def _clean_optional(value: Any) -> Optional[str]:
    text = _clean_title(value)
    return text or None


def _first_string(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if not value:
            return ""
        value = value[0]
    if value is None:
        return ""
    return str(value).strip()


def _first_int(value: Any) -> Optional[int]:
    values = _int_list(value)
    return values[0] if values else None


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            parsed = _single_int(item)
            if parsed is not None:
                result.append(parsed)
        return result

    parsed = _single_int(value)
    return [parsed] if parsed is not None else []


def _single_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
