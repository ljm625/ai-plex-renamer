from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from .debug import DebugLogger, debug_event
from .http_client import proxy_debug_info, urlopen_with_environment_proxy
from .models import MediaGuess


TMDBTransport = Callable[[str, Mapping[str, str], int], Mapping[str, Any]]
DEFAULT_TMDB_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_TMDB_RETRY_ATTEMPTS = 2
DEFAULT_TMDB_RETRY_DELAY_SECONDS = 0.5
TMDB_CACHE_VERSION = 1


class TMDBHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        message = f"HTTP {status_code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class TMDBTransportError(RuntimeError):
    pass


class TMDBClient:
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        language: str = "en-US",
        region: Optional[str] = None,
        include_adult: bool = False,
        timeout: int = 20,
        base_url: str = "https://api.themoviedb.org/3",
        transport: Optional[TMDBTransport] = None,
        debug: Optional[DebugLogger] = None,
        cache_path: Optional[Path] = None,
        cache_ttl_seconds: Optional[int] = DEFAULT_TMDB_CACHE_TTL_SECONDS,
        retry_attempts: int = DEFAULT_TMDB_RETRY_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_TMDB_RETRY_DELAY_SECONDS,
    ) -> None:
        self.bearer_token = bearer_token
        self.api_key = api_key
        self.language = language
        self.region = region
        self.include_adult = include_adult
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _urlopen_json
        self._debug = debug
        self.cache_path = Path(cache_path).expanduser() if cache_path is not None else None
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, Mapping[str, Any]] = {}
        self._disk_cache: dict[str, Any] = {}
        self._disk_cache_loaded = False
        self.retry_attempts = max(0, retry_attempts)
        self.retry_delay_seconds = max(0.0, retry_delay_seconds)

    @classmethod
    def from_environment(
        cls,
        bearer_token: Optional[str] = None,
        api_key: Optional[str] = None,
        language: str = "en-US",
        region: Optional[str] = None,
        include_adult: bool = False,
        timeout: int = 20,
        debug: Optional[DebugLogger] = None,
        cache_path: Optional[Path] = None,
        cache_enabled: bool = True,
        cache_ttl_seconds: Optional[int] = DEFAULT_TMDB_CACHE_TTL_SECONDS,
        retry_attempts: int = DEFAULT_TMDB_RETRY_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_TMDB_RETRY_DELAY_SECONDS,
    ) -> Optional["TMDBClient"]:
        token = bearer_token or os.getenv("TMDB_BEARER_TOKEN")
        key = api_key or os.getenv("TMDB_API_KEY")
        if not token and not key:
            debug_event(debug, "tmdb disabled", {"reason": "No TMDB_BEARER_TOKEN or TMDB_API_KEY configured."})
            return None
        resolved_cache_path = None
        if cache_enabled:
            env_cache_path = os.getenv("TMDB_CACHE_PATH")
            resolved_cache_path = cache_path or (Path(env_cache_path) if env_cache_path else _default_cache_path())
        return cls(
            bearer_token=token,
            api_key=key,
            language=language,
            region=region,
            include_adult=include_adult,
            timeout=timeout,
            debug=debug,
            cache_path=resolved_cache_path,
            cache_ttl_seconds=cache_ttl_seconds,
            retry_attempts=retry_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    def enrich(self, guess: MediaGuess) -> MediaGuess:
        if not guess.is_usable():
            return guess

        try:
            if guess.media_type == "movie":
                return self._enrich_movie(guess)
            if guess.media_type == "tv":
                return self._enrich_tv(guess)
        except Exception as exc:
            return _append_reason(guess, f"TMDB lookup failed: {exc}")
        return guess

    def _enrich_movie(self, guess: MediaGuess) -> MediaGuess:
        result = self._best_result(
            "/search/movie",
            {
                "query": guess.title,
                "include_adult": _bool_param(self.include_adult),
                "primary_release_year": guess.year,
            },
            guess.year,
        )
        if not result:
            return guess

        title = _first_text(result, "title", "original_title") or guess.title
        year = _year_from_date(result.get("release_date")) or guess.year
        return replace(
            guess,
            title=title,
            year=year,
            confidence=max(guess.confidence, 0.92),
            reason=_join_reasons(guess.reason, "TMDB matched movie title/year."),
        )

    def _enrich_tv(self, guess: MediaGuess) -> MediaGuess:
        result = self._best_result(
            "/search/tv",
            {
                "query": guess.title,
                "include_adult": _bool_param(self.include_adult),
                "first_air_date_year": guess.year,
            },
            guess.year,
        )
        if not result:
            return guess

        title = _first_text(result, "name", "original_name") or guess.title
        year = _year_from_date(result.get("first_air_date")) or guess.year
        episode_title = guess.episode_title
        series_id = result.get("id")
        episode_reason = ""
        if series_id and guess.season is not None and guess.episode is not None and guess.episode_end is None:
            try:
                episode_title = self._episode_title(series_id, guess.season, guess.episode) or episode_title
            except Exception as exc:
                episode_reason = f"TMDB episode title lookup failed: {exc}"

        return replace(
            guess,
            title=title,
            year=year,
            episode_title=episode_title,
            confidence=max(guess.confidence, 0.92),
            reason=_join_reasons(guess.reason, "TMDB matched TV title/year.", episode_reason),
        )

    def _episode_title(self, series_id: Any, season: int, episode: int) -> Optional[str]:
        data = self._request(
            f"/tv/{series_id}/season/{season}/episode/{episode}",
            {},
        )
        return _first_text(data, "name")

    def _best_result(
        self,
        endpoint: str,
        params: Mapping[str, Any],
        expected_year: Optional[int],
    ) -> Optional[Mapping[str, Any]]:
        data = self._request(endpoint, params)
        results = data.get("results") if isinstance(data, Mapping) else None
        if not isinstance(results, list) or not results:
            return None

        scored = []
        for index, result in enumerate(results):
            if not isinstance(result, Mapping):
                continue
            year = _year_from_date(result.get("release_date") or result.get("first_air_date"))
            score = 0
            if expected_year is not None and year == expected_year:
                score += 100
            if result.get("popularity") is not None:
                try:
                    score += min(float(result["popularity"]), 50.0)
                except (TypeError, ValueError):
                    pass
            scored.append((score, -index, result))

        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][2]

    def _request(self, endpoint: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        clean_params = {
            key: _normalize_param(value)
            for key, value in params.items()
            if value is not None and value != ""
        }
        clean_params.setdefault("language", self.language)
        if self.region:
            clean_params.setdefault("region", self.region)
        if self.api_key:
            clean_params.setdefault("api_key", self.api_key)

        cache_key = _request_cache_key(endpoint, clean_params)
        if cache_key in self._cache:
            debug_event(
                self._debug,
                "tmdb cache hit",
                {
                    "source": "memory",
                    "endpoint": endpoint,
                    "params": clean_params,
                    "response": self._cache[cache_key],
                },
            )
            return self._cache[cache_key]

        cached_data = self._read_disk_cache(cache_key)
        if cached_data is not None:
            self._cache[cache_key] = cached_data
            debug_event(
                self._debug,
                "tmdb cache hit",
                {
                    "source": "disk",
                    "endpoint": endpoint,
                    "params": clean_params,
                    "response": cached_data,
                },
            )
            return cached_data

        url = f"{self.base_url}{endpoint}?{urllib.parse.urlencode(clean_params)}"
        headers = {"accept": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        debug_event(
            self._debug,
            "tmdb request",
            {
                "endpoint": endpoint,
                "url": url,
                "params": clean_params,
                "headers": headers,
                "timeout": self.timeout,
                "proxies": proxy_debug_info(),
            },
        )
        data = self._request_with_retries(endpoint, url, clean_params, headers)
        debug_event(self._debug, "tmdb response", {"endpoint": endpoint, "response": data})
        self._cache[cache_key] = data
        self._write_disk_cache(cache_key, data)
        return data

    def _request_with_retries(
        self,
        endpoint: str,
        url: str,
        params: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        for attempt in range(self.retry_attempts + 1):
            try:
                return self._transport(url, headers, self.timeout)
            except Exception as exc:
                if attempt >= self.retry_attempts or not _is_retryable_error(exc):
                    raise
                delay = self.retry_delay_seconds * (attempt + 1)
                debug_event(
                    self._debug,
                    "tmdb retry",
                    {
                        "endpoint": endpoint,
                        "params": params,
                        "attempt": attempt + 1,
                        "max_attempts": self.retry_attempts,
                        "delay_seconds": delay,
                        "error": str(exc),
                    },
                )
                if delay > 0:
                    time.sleep(delay)
        raise RuntimeError("TMDB retry loop exited unexpectedly.")

    def _read_disk_cache(self, cache_key: str) -> Optional[Mapping[str, Any]]:
        self._load_disk_cache()
        entry = self._disk_cache.get(cache_key)
        if not isinstance(entry, Mapping):
            return None
        created_at = entry.get("created_at")
        if self._is_cache_entry_expired(created_at):
            return None
        response = entry.get("response")
        return response if isinstance(response, Mapping) else None

    def _write_disk_cache(self, cache_key: str, data: Mapping[str, Any]) -> None:
        if self.cache_path is None:
            return
        self._load_disk_cache()
        now = int(time.time())
        self._disk_cache[cache_key] = {
            "created_at": now,
            "response": data,
        }
        self._prune_expired_disk_cache(now)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": TMDB_CACHE_VERSION,
                "entries": self._disk_cache,
            }
            temp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(self.cache_path)
        except OSError as exc:
            debug_event(self._debug, "tmdb cache write failed", {"path": self.cache_path, "error": str(exc)})

    def _load_disk_cache(self) -> None:
        if self._disk_cache_loaded:
            return
        self._disk_cache_loaded = True
        if self.cache_path is None:
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            debug_event(self._debug, "tmdb cache read failed", {"path": self.cache_path, "error": str(exc)})
            return
        if not isinstance(payload, Mapping) or payload.get("version") != TMDB_CACHE_VERSION:
            return
        entries = payload.get("entries")
        if isinstance(entries, Mapping):
            self._disk_cache = dict(entries)

    def _is_cache_entry_expired(self, created_at: Any) -> bool:
        if self.cache_ttl_seconds is None:
            return False
        if not isinstance(created_at, (int, float)):
            return True
        return time.time() - float(created_at) > self.cache_ttl_seconds

    def _prune_expired_disk_cache(self, now: int) -> None:
        if self.cache_ttl_seconds is None:
            return
        expired_keys = [
            key
            for key, entry in self._disk_cache.items()
            if not isinstance(entry, Mapping)
            or not isinstance(entry.get("created_at"), (int, float))
            or now - float(entry["created_at"]) > self.cache_ttl_seconds
        ]
        for key in expired_keys:
            self._disk_cache.pop(key, None)


def _urlopen_json(url: str, headers: Mapping[str, str], timeout: int) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urlopen_with_environment_proxy(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TMDBHTTPError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise TMDBTransportError(str(exc.reason)) from exc
    except (OSError, TimeoutError) as exc:
        raise TMDBTransportError(str(exc)) from exc

    data = json.loads(payload)
    if not isinstance(data, Mapping):
        raise ValueError("TMDB response must be a JSON object.")
    return data


def _first_text(data: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _year_from_date(value: Any) -> Optional[int]:
    if not isinstance(value, str) or len(value) < 4:
        return None
    try:
        return int(value[:4])
    except ValueError:
        return None


def _append_reason(guess: MediaGuess, reason: str) -> MediaGuess:
    return replace(guess, reason=_join_reasons(guess.reason, reason))


def _join_reasons(*reasons: str) -> str:
    return " ".join(reason.strip() for reason in reasons if reason and reason.strip())


def _normalize_param(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    return value


def _request_cache_key(endpoint: str, params: Mapping[str, Any]) -> str:
    cache_params = {
        key: value
        for key, value in params.items()
        if key.lower() != "api_key"
    }
    payload = {
        "endpoint": endpoint,
        "params": sorted(cache_params.items()),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, TMDBHTTPError):
        return exc.status_code == 408 or exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, (TMDBTransportError, TimeoutError, OSError)):
        return True
    if isinstance(exc, RuntimeError):
        text = str(exc).strip().lower()
        if text.startswith("http 4") and not text.startswith(("http 408", "http 429")):
            return False
        return True
    return False


def _default_cache_path() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ai-plex-renamer" / "tmdb-cache.json"


def _bool_param(value: bool) -> str:
    return "true" if value else "false"
