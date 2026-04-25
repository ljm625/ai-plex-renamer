from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


DebugLogger = Callable[[str], None]

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "nvidia_api_key",
    "tmdb_api_key",
    "tmdb_token",
}


def stderr_logger(message: str) -> None:
    print(message, file=sys.stderr)


def debug_event(debug: Optional[DebugLogger], title: str, data: Any = None) -> None:
    if debug is None:
        return
    if data is None:
        debug(f"[verbose] {title}")
        return
    payload = json.dumps(redact(data), ensure_ascii=False, indent=2, default=str)
    debug(f"[verbose] {title}\n{payload}")


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redacted_value(str(key), item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return [redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _redacted_value(key: str, value: Any) -> Any:
    normalized = key.lower().replace("-", "_")
    if normalized in SENSITIVE_KEYS:
        return "***REDACTED***"
    if normalized == "url" and isinstance(value, str):
        return _redact_url(value)
    if normalized == "headers" and isinstance(value, Mapping):
        return {
            str(header): "***REDACTED***" if str(header).lower() == "authorization" else redact(item)
            for header, item in value.items()
        }
    if isinstance(value, str) and value.lower().startswith("bearer "):
        return "Bearer ***REDACTED***"
    return redact(value)


def _redact_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = [
        (key, "***REDACTED***" if key.lower() in SENSITIVE_KEYS else value)
        for key, value in query
    ]
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urllib.parse.urlencode(redacted_query, safe="*"),
            parsed.fragment,
        )
    )
