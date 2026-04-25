from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .debug import DebugLogger, debug_event
from .http_client import proxy_debug_info, urlopen_with_environment_proxy
from .models import MediaGuess


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

AITransport = Callable[[str, Mapping[str, str], Mapping[str, Any], int], Mapping[str, Any]]

SINGLE_RESPONSE_SCHEMA = """{
  "media_type": "tv" | "movie" | "unknown",
  "title": "canonical show or movie title, no release tags",
  "year": 2024 | null,
  "season": 1 | null,
  "episode": 2 | null,
  "episode_end": 3 | null,
  "episode_title": "episode title if clear" | null,
  "confidence": 0.0,
  "reason": "short reason"
}"""

BATCH_RESPONSE_SCHEMA = """{
  "files": [
    {
      "index": 1,
      "media_type": "tv" | "movie" | "unknown",
      "title": "canonical show or movie title, no release tags",
      "year": 2024 | null,
      "season": 1 | null,
      "episode": 2 | null,
      "episode_end": 3 | null,
      "episode_title": "episode title if clear" | null,
      "confidence": 0.0,
      "reason": "short reason"
    }
  ]
}"""


class AIUnavailable(RuntimeError):
    pass


class NvidiaAIClassifier:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 600,
        temperature: float = 0.0,
        max_tokens: int = 10240,
        min_interval: float = 0.0,
        json_repair_attempts: int = 1,
        transport: Optional[AITransport] = None,
        debug: Optional[DebugLogger] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise AIUnavailable("NVIDIA_API_KEY is not set. Create a key at build.nvidia.com.")

        self.model = model or os.getenv("NVIDIA_MODEL") or NVIDIA_DEFAULT_MODEL
        self.base_url = (base_url or os.getenv("NVIDIA_BASE_URL") or NVIDIA_DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = min_interval
        self.json_repair_attempts = max(0, json_repair_attempts)
        self._transport = transport or _urlopen_json
        self._debug = debug
        self._last_call = 0.0

    def classify(
        self,
        path: Path,
        root: Path,
        library_hint: str,
        local_guess: MediaGuess,
    ) -> MediaGuess:
        prompt = build_prompt(path, root, library_hint, local_guess)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You identify media filenames for Plex and return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        content = self._chat(payload, self._headers(), "nvidia request", "nvidia response")
        return self._parse_with_json_repair(
            content,
            parse_ai_response,
            original_prompt=prompt,
            expected_schema=SINGLE_RESPONSE_SCHEMA,
            max_tokens=self.max_tokens,
        )

    def classify_many(
        self,
        paths: list[Path],
        root: Path,
        library_hint: str,
        local_guesses: Mapping[Path, MediaGuess],
    ) -> dict[Path, MediaGuess]:
        if not paths:
            return {}

        prompt = build_batch_prompt(paths, root, library_hint, local_guesses)
        max_tokens = max(self.max_tokens, min(4096, 360 * len(paths)))
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You identify related media filenames for Plex and return JSON only.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        content = self._chat(
            payload,
            self._headers(),
            "nvidia batch request",
            "nvidia batch response",
            {"file_count": len(paths)},
        )
        return self._parse_with_json_repair(
            content,
            lambda value: parse_ai_batch_response(value, paths),
            original_prompt=prompt,
            expected_schema=BATCH_RESPONSE_SCHEMA,
            max_tokens=max_tokens,
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _chat(
        self,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
        request_title: str,
        response_title: str,
        extra_debug: Optional[Mapping[str, Any]] = None,
    ) -> str:
        self._wait_for_rate_limit()
        debug_data: dict[str, Any] = {
            "url": f"{self.base_url}/chat/completions",
            "headers": headers,
            "payload": payload,
            "timeout": self.timeout,
            "proxies": proxy_debug_info(),
        }
        if extra_debug:
            debug_data.update(extra_debug)
        debug_event(self._debug, request_title, debug_data)
        response = self._transport(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            self.timeout,
        )
        debug_event(self._debug, response_title, response)
        self._last_call = time.monotonic()
        return _message_content(response)

    def _parse_with_json_repair(
        self,
        content: str,
        parser: Callable[[str], Any],
        original_prompt: str,
        expected_schema: str,
        max_tokens: int,
    ) -> Any:
        try:
            return parser(content)
        except Exception as exc:
            parse_error = exc

        for attempt in range(1, self.json_repair_attempts + 1):
            repair_payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You repair invalid JSON. Return valid JSON only.",
                    },
                    {
                        "role": "user",
                        "content": build_json_repair_prompt(
                            invalid_response=content,
                            error=parse_error,
                            original_prompt=original_prompt,
                            expected_schema=expected_schema,
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": max(max_tokens, self.max_tokens, 1024),
                "stream": False,
            }
            content = self._chat(
                repair_payload,
                self._headers(),
                "nvidia json repair request",
                "nvidia json repair response",
                {"attempt": attempt},
            )
            try:
                return parser(content)
            except Exception as exc:
                parse_error = exc

        raise ValueError(
            f"AI returned invalid JSON after {self.json_repair_attempts} repair attempt(s): {parse_error}"
        ) from parse_error

    def _wait_for_rate_limit(self) -> None:
        if self._last_call <= 0 or self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call
        remaining = self.min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)


def build_prompt(path: Path, root: Path, library_hint: str, local_guess: MediaGuess) -> str:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        relative_path = path

    local_guess_json = json.dumps(local_guess.to_dict(), ensure_ascii=False)
    return f"""Analyze this media file path and return ONLY one compact JSON object.

Library hint: {library_hint}
Relative path: {relative_path}
File name: {path.name}
Parent folder: {path.parent.name}
Local parser guess: {local_guess_json}

Return this exact schema:
{{
  "media_type": "tv" | "movie" | "unknown",
  "title": "canonical show or movie title, no release tags",
  "year": 2024 | null,
  "season": 1 | null,
  "episode": 2 | null,
  "episode_end": 3 | null,
  "episode_title": "episode title if clear" | null,
  "confidence": 0.0,
  "reason": "short reason"
}}

Rules:
- For TV, identify the show title, season number, episode number, optional ending episode for multi-episode files, and optional episode title.
- For TV specials/extras in folders such as SP, SPs, Specials, or Season 00, use season 0. For markers like CM04, SP04, NCOP01, or NCED01, use the marker number as the episode. If a special such as Menu or Commentary has no safe episode number, return media_type "unknown".
- If a file sits inside a folder that looks like a title and the filename has 前編, 後編, 上巻, 下巻, Part 1, or Part 2, prefer TV episode metadata.
- For TV, include the show release year only when the file or folder makes it clear.
- For movies, identify the movie title and release year when the file or folder makes it clear.
- Do not invent a movie year if it is not present or strongly implied.
- Strip release tags such as resolution, source, codec, audio, group names, fansub tags, and language tags.
- Use the original title language when possible.
- If you are not confident enough to rename safely, use media_type "unknown".
- Output JSON only; no Markdown and no explanation outside JSON.
"""


def build_batch_prompt(
    paths: list[Path],
    root: Path,
    library_hint: str,
    local_guesses: Mapping[Path, MediaGuess],
) -> str:
    files = []
    for index, path in enumerate(paths, start=1):
        try:
            relative_path = str(path.relative_to(root))
        except ValueError:
            relative_path = str(path)
        local_guess = local_guesses.get(path, MediaGuess.unknown("No local guess."))
        files.append(
            {
                "index": index,
                "relative_path": relative_path,
                "file_name": path.name,
                "parent_folder": path.parent.name,
                "local_guess": local_guess.to_dict(),
            }
        )

    files_json = json.dumps(files, ensure_ascii=False, indent=2)
    return f"""Analyze this group of media files from the same folder and return ONLY one compact JSON object.

Library hint: {library_hint}
Files:
{files_json}

Return this exact schema:
{{
  "files": [
    {{
      "index": 1,
      "media_type": "tv" | "movie" | "unknown",
      "title": "canonical show or movie title, no release tags",
      "year": 2024 | null,
      "season": 1 | null,
      "episode": 2 | null,
      "episode_end": 3 | null,
      "episode_title": "episode title if clear" | null,
      "confidence": 0.0,
      "reason": "short reason"
    }}
  ]
}}

Rules:
- Treat the files as related sibling files. Use the whole list to infer shared show/movie title and episode numbering.
- For TV specials/extras in folders such as SP, SPs, Specials, or Season 00, use season 0. For markers like CM04, SP04, NCOP01, or NCED01, use the marker number as the episode. If a special such as Menu or Commentary has no safe episode number, return media_type "unknown".
- If filenames differ mostly by #1, ＃2, 第1話, episode numbers, 前編/後編, Part 1/Part 2, prefer TV episode metadata.
- For TV, keep the same show title across related files unless the folder clearly contains multiple shows.
- For TV, include the show release year only when the file or folder makes it clear.
- For movies, identify the movie title and release year when clear.
- Strip release tags such as resolution, source, codec, audio, group names, fansub tags, and language tags.
- Use the original title language when possible.
- If a file is unsafe to rename, use media_type "unknown" for that file.
- Return one result for every input index. Output JSON only; no Markdown and no explanation outside JSON.
"""


def build_json_repair_prompt(
    invalid_response: str,
    error: Exception,
    original_prompt: str,
    expected_schema: str,
) -> str:
    return f"""The previous response could not be parsed as valid JSON.

Parser/schema error:
{error}

Original task:
{original_prompt}

Expected JSON schema:
{expected_schema}

Invalid response:
```text
{invalid_response}
```

Repair rules:
- Return only one valid JSON value matching the expected schema.
- Use double-quoted JSON strings, explicit commas, and null instead of Python None.
- Do not wrap the answer in Markdown.
- Preserve the intended media metadata from the invalid response when possible.
- If a field cannot be recovered safely, use null, an empty string, or media_type "unknown".
"""


def parse_ai_response(response: str) -> MediaGuess:
    data = _loads_json_object(response)
    return MediaGuess.from_dict(data)


def parse_ai_batch_response(response: str, paths: list[Path]) -> dict[Path, MediaGuess]:
    data = _loads_json_value(response)
    raw_files = data.get("files") if isinstance(data, Mapping) else None
    if raw_files is None and isinstance(data, list):
        raw_files = data
    if not isinstance(raw_files, list):
        raise ValueError("AI batch response JSON must include a files array.")

    by_path: dict[Path, MediaGuess] = {}
    for item in raw_files:
        if not isinstance(item, Mapping):
            continue
        index = _optional_index(item.get("index"), len(paths))
        if index is None:
            continue
        by_path[paths[index - 1]] = MediaGuess.from_dict(dict(item))
    return by_path


def _message_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("NVIDIA response did not include choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("NVIDIA response choice must be an object.")
    message = first_choice.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first_choice.get("text")
    if isinstance(text, str):
        return text
    raise ValueError("NVIDIA response did not include message content.")


def _loads_json_object(response: str) -> dict:
    data = _loads_json_value(response)
    if not isinstance(data, dict):
        raise ValueError("AI response JSON must be an object.")
    return data


def _loads_json_value(response: str) -> Any:
    text = response.strip()
    block_match = JSON_BLOCK_PATTERN.search(text)
    if block_match:
        text = block_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"AI response did not contain a JSON object: {response!r}")
        data = json.loads(text[start : end + 1])

    return data


def _optional_index(value: Any, max_index: int) -> Optional[int]:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    if 1 <= index <= max_index:
        return index
    return None


def _urlopen_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: int,
) -> Mapping[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen_with_environment_proxy(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NVIDIA API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NVIDIA API request failed: {exc.reason}") from exc

    data = json.loads(response_body)
    if not isinstance(data, Mapping):
        raise ValueError("NVIDIA response must be a JSON object.")
    return data
