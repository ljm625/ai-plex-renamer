from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .ai import NVIDIA_DEFAULT_BASE_URL, NVIDIA_DEFAULT_MODEL, NvidiaAIClassifier
from .debug import stderr_logger
from .models import RenamePlan
from .renamer import apply_plan, build_plans, iter_media_files
from .tmdb import TMDBClient


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.path).expanduser().resolve()

    if not root.exists():
        parser.error(f"Path does not exist: {root}")

    debug = stderr_logger if args.verbose else None
    classifier = None
    if not args.no_ai:
        classifier = LazyNvidiaAIClassifier(
            api_key=args.nvidia_api_key,
            model=args.nvidia_model,
            base_url=args.nvidia_base_url,
            timeout=args.ai_timeout,
            temperature=args.ai_temperature,
            max_tokens=args.ai_max_tokens,
            min_interval=args.ai_min_interval,
            debug=debug,
        )
    tmdb_client = None
    if not args.no_tmdb:
        tmdb_client = TMDBClient.from_environment(
            bearer_token=args.tmdb_token,
            api_key=args.tmdb_api_key,
            language=args.tmdb_language,
            region=args.tmdb_region,
            include_adult=args.tmdb_include_adult,
            timeout=args.tmdb_timeout,
            debug=debug,
        )

    files = list(iter_media_files(root, recursive=args.recursive, include_hidden=args.include_hidden))
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print("No video files found.")
        return 0

    plans = build_plans(
        files,
        root=root if root.is_dir() else root.parent,
        classifier=classifier,
        tmdb_client=tmdb_client,
        library_hint=args.type,
        collision=args.collision,
        debug=debug,
    )

    final_plans: list[RenamePlan] = []
    for index, plan in enumerate(plans, start=1):
        if not args.json:
            print(f"[{index}/{len(plans)}] {plan.source}")
        if args.apply:
            plan = apply_plan(plan)
        final_plans.append(plan)
        if not args.json:
            print(format_plan(plan, apply=args.apply))

    if args.json:
        print(json.dumps([plan.to_dict() for plan in final_plans], ensure_ascii=False, indent=2))
    else:
        print_summary(final_plans, applied=args.apply)
    return 1 if any(plan.status == "error" for plan in final_plans) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-plex-renamer",
        description="Rename media files to Plex-friendly names with GuessIt, TMDB, and NVIDIA AI fallback.",
    )
    parser.add_argument("path", help="Media file or folder to scan.")
    parser.add_argument(
        "--type",
        choices=["auto", "tv", "movie"],
        default="auto",
        help="Media library hint. Default: auto.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rename files. Without this flag the command only previews changes.",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable NVIDIA AI fallback; use GuessIt, TMDB, and local heuristics only.",
    )
    parser.add_argument(
        "--no-tmdb",
        action="store_true",
        help="Disable TMDB enrichment even when TMDB credentials are configured.",
    )
    parser.add_argument(
        "--tmdb-token",
        default=None,
        help="TMDB API Read Access Token. Defaults to TMDB_BEARER_TOKEN.",
    )
    parser.add_argument(
        "--tmdb-api-key",
        default=None,
        help="TMDB v3 API key. Defaults to TMDB_API_KEY.",
    )
    parser.add_argument(
        "--tmdb-language",
        default="en-US",
        help="TMDB response language. Default: en-US.",
    )
    parser.add_argument(
        "--tmdb-region",
        default=None,
        help="Optional TMDB region filter such as US or CN.",
    )
    parser.add_argument(
        "--tmdb-include-adult",
        action="store_true",
        help="Include adult TMDB results. Useful when your library contains adult OVA/anime entries.",
    )
    parser.add_argument(
        "--tmdb-timeout",
        type=int,
        default=20,
        help="TMDB request timeout in seconds. Default: 20.",
    )
    parser.add_argument(
        "--nvidia-api-key",
        default=None,
        help="NVIDIA API key. Defaults to NVIDIA_API_KEY.",
    )
    parser.add_argument(
        "--nvidia-base-url",
        default=None,
        help=f"NVIDIA OpenAI-compatible base URL. Defaults to NVIDIA_BASE_URL or {NVIDIA_DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--nvidia-model",
        "--model",
        dest="nvidia_model",
        default=None,
        help=f"NVIDIA model name. Defaults to NVIDIA_MODEL or {NVIDIA_DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--ai-timeout",
        "--timeout",
        dest="ai_timeout",
        type=int,
        default=45,
        help="AI request timeout in seconds.",
    )
    parser.add_argument("--ai-temperature", type=float, default=0.0, help="AI sampling temperature. Default: 0.0.")
    parser.add_argument("--ai-max-tokens", type=int, default=512, help="Maximum AI response tokens. Default: 512.")
    parser.add_argument(
        "--ai-min-interval",
        "--min-interval",
        dest="ai_min_interval",
        type=float,
        default=0.0,
        help="Minimum seconds between AI requests. Default: 0.",
    )
    parser.add_argument(
        "--collision",
        choices=["skip", "suffix"],
        default="skip",
        help="What to do when the target filename already exists. Default: skip.",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan folders recursively. Default: true.",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and files inside hidden directories.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N video files.")
    parser.add_argument("--json", action="store_true", help="Print the rename plan as JSON.")
    parser.add_argument("--verbose", action="store_true", help="Print local, TMDB, and AI debug details to stderr.")
    return parser


class LazyNvidiaAIClassifier:
    def __init__(
        self,
        api_key: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
        timeout: int,
        temperature: float,
        max_tokens: int,
        min_interval: float,
        debug,
    ) -> None:
        self._kwargs = {
            "api_key": api_key,
            "model": model,
            "base_url": base_url,
            "timeout": timeout,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "min_interval": min_interval,
            "debug": debug,
        }
        self._classifier: Optional[NvidiaAIClassifier] = None

    def classify(self, *args, **kwargs):
        if self._classifier is None:
            self._classifier = NvidiaAIClassifier(**self._kwargs)
        return self._classifier.classify(*args, **kwargs)

    def classify_many(self, *args, **kwargs):
        if self._classifier is None:
            self._classifier = NvidiaAIClassifier(**self._kwargs)
        return self._classifier.classify_many(*args, **kwargs)


def format_plan(plan: RenamePlan, apply: bool) -> str:
    if plan.status in {"planned", "renamed", "unchanged"} and plan.target:
        verb = "RENAMED" if plan.status == "renamed" else "DRY" if not apply else plan.status.upper()
        return (
            f"  {verb}: {plan.source.name} -> {plan.target.name}\n"
            f"  guess: {plan.guess.media_type}, confidence={plan.guess.confidence:.2f}, "
            f"reason={plan.guess.reason}"
        )
    return (
        f"  {plan.status.upper()}: {plan.source.name}\n"
        f"  reason: {plan.message or plan.guess.reason}"
    )


def print_summary(plans: Iterable[RenamePlan], applied: bool) -> None:
    counts: dict[str, int] = {}
    total = 0
    for plan in plans:
        total += 1
        counts[plan.status] = counts.get(plan.status, 0) + 1
    mode = "apply" if applied else "dry-run"
    parts = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
    print(f"\nSummary ({mode}): total={total}, {parts}")
    if not applied:
        print("Run again with --apply to rename the planned files.")


if __name__ == "__main__":
    raise SystemExit(main())
