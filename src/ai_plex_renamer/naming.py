from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .models import MediaGuess


VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".iso",
    ".m2ts",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".webm",
    ".wmv",
}

INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
WHITESPACE = re.compile(r"\s+")


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def sanitize_component(value: str, fallback: str = "Unknown") -> str:
    cleaned = INVALID_FILENAME_CHARS.sub(" ", value)
    cleaned = WHITESPACE.sub(" ", cleaned).strip(" .")
    return cleaned or fallback


def build_plex_filename(guess: MediaGuess, original_extension: str) -> str:
    extension = original_extension if original_extension.startswith(".") else f".{original_extension}"
    if guess.media_type == "tv":
        return _build_tv_filename(guess, extension)
    if guess.media_type == "movie":
        return _build_movie_filename(guess, extension)
    raise ValueError("Cannot build a Plex name for an unknown media type.")


def build_plex_folder_name(guess: MediaGuess) -> str:
    title = sanitize_component(guess.title, fallback="Unknown")
    if guess.year and not title.endswith(f"({guess.year})"):
        return f"{title} ({guess.year})"
    return title


def resolve_collision(target: Path, source: Path, strategy: str) -> Path:
    if target == source or not target.exists():
        return target
    if strategy == "skip":
        raise FileExistsError(f"Target already exists: {target}")
    if strategy != "suffix":
        raise ValueError(f"Unsupported collision strategy: {strategy}")

    for index in range(1, 10_000):
        candidate = target.with_name(f"{target.stem} ({index}){target.suffix}")
        if candidate == source or not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find a free target name for: {target}")


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _build_tv_filename(guess: MediaGuess, extension: str) -> str:
    if guess.season is None or guess.episode is None:
        raise ValueError("TV episodes need season and episode numbers.")

    title = sanitize_component(guess.title, fallback="Unknown Show")
    if guess.year and not title.endswith(f"({guess.year})"):
        title = f"{title} ({guess.year})"
    episode_code = f"S{guess.season:02d}E{guess.episode:02d}"
    if guess.episode_end is not None and guess.episode_end > guess.episode:
        episode_code = f"{episode_code}-E{guess.episode_end:02d}"

    parts = [title, episode_code]
    if guess.episode_title:
        episode_title = sanitize_component(guess.episode_title, fallback="")
        if episode_title and episode_title.lower() != title.lower():
            parts.append(episode_title)
    return " - ".join(parts) + extension


def _build_movie_filename(guess: MediaGuess, extension: str) -> str:
    title = sanitize_component(guess.title, fallback="Unknown Movie")
    if guess.year:
        return f"{title} ({guess.year}){extension}"
    return f"{title}{extension}"
