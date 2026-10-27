"""Command line: ``course-assistant ingest | ideas | serve | eval``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import textwrap
from pathlib import Path

from .config import Settings, load_dotenv_if_present
from .errors import IndexModelMismatchError, ProviderError
from .models import FORMATS


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # HTTP client debug logs can include request metadata; keep them quiet.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


EMPTY_MESSAGES = {
    "empty_index": "The index is empty: run `course-assistant ingest` first.",
    "no_topic_match": "Nothing in your lessons matches that topic. Try other words.",
    "all_too_short": "Every passage is shorter than MIN_PASSAGE_CHARS: lower it or add content.",
    "topic_matches_used": (
        "The passages that match this topic were used recently or are too short. "
        "Try another topic or lower SKIP_USED_DAYS."
    ),
    "all_recently_used": (
        "Every passage was used within the last SKIP_USED_DAYS days: wait, lower it, "
        "or try --topic."
    ),
    "model_returned_none": "The model's ideas all failed the checks. Try again.",
}


def _cmd_ingest(settings: Settings, args: argparse.Namespace) -> int:
    from .services import build_ingestor

    report = build_ingestor(settings).run(rebuild=args.rebuild)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def _cmd_ideas(settings: Settings, args: argparse.Namespace) -> int:
    from .errors import GenerationError
    from .models import format_label
    from .services import build_services

    services = build_services(settings)
    if services.store.stats()["chunks"] == 0:
        print("The index is empty: run `course-assistant ingest` first.", file=sys.stderr)
        return 2
    try:
        result = services.ideas.suggest(args.count, fmt=args.format, topic=args.topic)
    except (ProviderError, GenerationError, ValueError) as exc:
        print(f"Could not make ideas: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if not result.ideas:
        print(EMPTY_MESSAGES.get(result.empty_reason or "", "No ideas this time."))
    for idea in result.ideas:
        print(f"#{idea.id} [{format_label(idea.format)}] {idea.title}")
        print(textwrap.indent(textwrap.fill(idea.hook, width=96), "    "))
        print(f"    from: {idea.doc_title} > {idea.heading}")
    if result.dropped:
        reasons = ", ".join(sorted({d.reason for d in result.dropped}))
        print(f"\n{len(result.dropped)} idea(s) dropped by the checks ({reasons}).")
    usage = result.usage
    cost = f"${usage.cost_usd:.5f}" if usage.cost_usd is not None else "n/a"
    print(
        f"\n{len(result.ideas)} kept of {result.requested} requested · "
        f"{result.fresh_passages_left} fresh passages · "
        f"{result.latency_ms.get('total', 0):.0f} ms · "
        f"{usage.input_tokens}+{usage.output_tokens} tokens · {cost}"
    )
    return 0


def _cmd_eval(settings: Settings, args: argparse.Namespace) -> int:
    from .errors import GenerationError
    from .eval import run_on_copy, write_reports

    try:
        report = run_on_copy(settings, rounds=args.rounds)
    except (FileNotFoundError, ProviderError, GenerationError) as exc:
        print(f"Eval failed: {exc}", file=sys.stderr)
        return 1
    md_path, json_path = write_reports(report, Path(args.out))
    print(f"Report written to {md_path} and {json_path}")
    return 0


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="course-assistant", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="index new or changed documents")
    p_ingest.add_argument("--rebuild", action="store_true", help="re-process every document")

    p_ideas = sub.add_parser("ideas", help="suggest content ideas from the terminal")
    p_ideas.add_argument("--count", type=int, default=None, help="how many (default from settings)")
    p_ideas.add_argument("--format", choices=sorted(FORMATS), default=None)
    p_ideas.add_argument("--topic", default=None)
    p_ideas.add_argument("--json", action="store_true", help="print the full structured result")

    p_serve = sub.add_parser("serve", help="run the API and chat UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_eval = sub.add_parser("eval", help="live check of the pipeline on a copy of the index")
    p_eval.add_argument("--rounds", type=int, default=3)
    p_eval.add_argument("--out", default="eval", help="folder for report.md and report.json")

    args = parser.parse_args(argv)
    load_dotenv_if_present()
    _configure_logging(args.verbose)
    settings = Settings.from_env()
    handlers = {
        "ingest": _cmd_ingest,
        "ideas": _cmd_ideas,
        "serve": _cmd_serve,
        "eval": _cmd_eval,
    }
    try:
        return handlers[args.command](settings, args)
    except (ProviderError, IndexModelMismatchError) as exc:
        # Plain message, no traceback: a missing key or a stale index is not a crash.
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
