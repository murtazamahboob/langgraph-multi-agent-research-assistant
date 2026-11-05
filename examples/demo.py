"""Offline demo: run the full multi-agent pipeline with no API key.

Uses the deterministic mock LLM and mock web search, so it works with
no network access and no paid API key. Writes a real Markdown report to
``outputs/demo_report.md`` and prints a short summary.

Run it with::

    python examples/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the demo from the repository root without installing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.graph import run_research  # noqa: E402
from research_assistant.llm import MockLLM  # noqa: E402
from research_assistant.search import MockSearchClient  # noqa: E402

DEMO_QUERY = "What is LangGraph and how does it work?"


def main() -> int:
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_report.md"

    final_state = run_research(
        DEMO_QUERY, MockLLM(), MockSearchClient(), max_results=3, max_retries=1
    )
    output_path.write_text(final_state["report"], encoding="utf-8")

    print("LangGraph Multi-Agent Research Assistant — offline demo")
    print(f"Query:   {final_state['query']}")
    print(f"Plan:    {len(final_state['plan'])} sub-question(s)")
    for item in final_state["plan"]:
        print(f"  - {item}")
    print(f"Sources: {len(final_state['findings'])} finding(s)")
    print(f"Report:  {output_path}")
    print()
    print("--- report preview (first 12 lines) ---")
    print("\n".join(final_state["report"].splitlines()[:12]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
