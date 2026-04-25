from __future__ import annotations

import json
import os
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import replace
from typing import Any, Callable, Mapping, Optional

from .debug import DebugLogger, debug_event
from .models import MediaGuess


TMDBTransport = Callable[[str, Mapping[str, str], int], Mapping[str, Any]]


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
        self._cache: dict[tuple[str, tuple[tuple[str, Any], ...]], Mapping[str, Any]] = {}

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
    ) -> Optional["TMDBClient"]:
        token = bearer_token or os.getenv("TMDB_BEARER_TOKEN")
        key = api_key or os.getenv("TMDB_API_KEY")
        if not token and not key:
            debug_event(debug, "tmdb disabled", {"reason": "No TMDB_BEARER_TOKEN or TMDB_API_KEY configured."})
            return None
        return cls(
            bearer_token=token,
            api_key=key,
            language=language,
            region=region,
            include_adult=include_adult,
            timeout=timeout,
            debug=debug,
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
        if series_id and guess.season is not None and guess.episode is not None and guess.episode_end is None:
            episode_title = self._episode_title(series_id, guess.season, guess.episode) or episode_title

        return replace(
            guess,
            title=title,
            year=year,
            episode_title=episode_title,
            confidence=max(guess.confidence, 0.92),
            reason=_join_reasons(guess.reason, "TMDB matched TV title/year."),
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

        cache_key = (endpoint, tuple(sorted(clean_params.items())))
        if cache_key in self._cache:
            debug_event(
                self._debug,
                "tmdb cache hit",
                {
                    "endpoint": endpoint,
                    "params": clean_params,
                    "response": self._cache[cache_key],
                },
            )
            return self._cache[cache_key]

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
            },
        )
        data = self._transport(url, headers, self.timeout)
        debug_event(self._debug, "tmdb response", {"endpoint": endpoint, "response": data})
        self._cache[cache_key] = data
        return data


def _urlopen_json(url: str, headers: Mapping[str, str], timeout: int) -> Mapping[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc

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


def _bool_param(value: bool) -> str:
    return "true" if value else "false"
