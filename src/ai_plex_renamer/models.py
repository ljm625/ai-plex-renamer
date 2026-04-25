from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


VALID_MEDIA_TYPES = {"tv", "movie", "unknown"}


@dataclass(frozen=True)
class MediaGuess:
    media_type: str
    title: str = ""
    year: Optional[int] = None
    season: Optional[int] = None
    episode: Optional[int] = None
    episode_end: Optional[int] = None
    episode_title: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""

    @classmethod
    def unknown(cls, reason: str = "") -> "MediaGuess":
        return cls(media_type="unknown", reason=reason)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaGuess":
        media_type = str(data.get("media_type") or "unknown").strip().lower()
        media_type = {
            "episode": "tv",
            "film": "movie",
            "series": "tv",
            "show": "tv",
            "tv_show": "tv",
            "tv-series": "tv",
        }.get(media_type, media_type)
        if media_type not in VALID_MEDIA_TYPES:
            media_type = "unknown"

        return cls(
            media_type=media_type,
            title=_string_or_empty(data.get("title")),
            year=_optional_int(data.get("year")),
            season=_optional_int(data.get("season")),
            episode=_optional_int(data.get("episode")),
            episode_end=_optional_int(data.get("episode_end")),
            episode_title=_optional_string(data.get("episode_title")),
            confidence=_bounded_float(data.get("confidence"), default=0.0),
            reason=_string_or_empty(data.get("reason")),
        )

    def is_usable(self) -> bool:
        if self.media_type == "tv":
            return bool(self.title and self.season is not None and self.episode is not None)
        if self.media_type == "movie":
            return bool(self.title)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_type": self.media_type,
            "title": self.title,
            "year": self.year,
            "season": self.season,
            "episode": self.episode,
            "episode_end": self.episode_end,
            "episode_title": self.episode_title,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RenamePlan:
    source: Path
    target: Optional[Path]
    guess: MediaGuess
    status: str
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": str(self.source),
            "target": str(self.target) if self.target else None,
            "guess": self.guess.to_dict(),
            "status": self.status,
            "message": self.message,
        }


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value: Any) -> Optional[str]:
    text = _string_or_empty(value)
    return text or None


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _bounded_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, result))
