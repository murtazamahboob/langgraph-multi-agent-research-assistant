"""Command-line interface.

Examples::

    # Offline demo mode (default): mock LLM + mock search, no API key.
    python -m research_assistant.cli "What is LangGraph?"

    # Live web search, still with the mock LLM (no API key needed).
    python -m research_assistant.cli "What is LangGraph?" --live-search

    # Fully live: real search + real OpenAI model (needs OPENAI_API_KEY).
    python -m research_assistant.cli "What is LangGraph?" --live
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .graph import run_research
from .llm import get_llm
from .search import DuckDuckGoSearchClient, MockSearchClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description=(
            "Multi-agent AI research assistant (LangGraph): planner, "
            "researcher and writer agents produce a cited Markdown report."
        ),
    )
    parser.add_argument("query", help="Research question to investigate.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Where to write the Markdown report "
        "(default: outputs/<slug>.md).",
    )
    parser.add_argument(
        "--live-search",
        action="store_true",
        help="Use real DuckDuckGo web search instead of the offline mock.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real web search AND a real OpenAI model "
        "(requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Max search results per sub-question (default: 3).",
    )
    return parser


def slugify(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    slug = "-".join(part for part in slug.split("-") if part)
    return slug[:60] or "report"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()
    max_results = args.max_results or settings.max_results_per_query

    use_live_llm = args.live
    use_live_search = args.live or args.live_search
    llm = get_llm(use_mock=not use_live_llm, model=settings.model)
    search_client = (
        DuckDuckGoSearchClient() if use_live_search else MockSearchClient()
    )

    final_state = None
    try:
        final_state = run_research(
            args.query,
            llm,
            search_client,
            max_results=max_results,
            max_retries=settings.max_retries,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output) if args.output else output_dir / f"{slugify(args.query)}.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_state["report"], encoding="utf-8")

    print(f"Query:   {final_state['query']}")
    print(f"Plan:    {len(final_state['plan'])} sub-question(s)")
    print(f"Sources: {len(final_state['findings'])} finding(s)")
    print(f"Report:  {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
