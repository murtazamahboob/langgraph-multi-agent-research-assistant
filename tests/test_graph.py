"""End-to-end tests for the compiled LangGraph pipeline."""

from __future__ import annotations

import pytest

from research_assistant.graph import build_graph, run_research
from research_assistant.llm import MockLLM
from research_assistant.search import MockSearchClient


class EmptySearchClient:
    """A search backend that always returns nothing (simulates outage)."""

    def search(self, query: str, max_results: int = 5):
        return []


def test_run_research_offline_end_to_end(mock_llm, mock_search):
    final = run_research("What is LangGraph?", mock_llm, mock_search)
    assert final["query"] == "What is LangGraph?"
    assert len(final["plan"]) == 3
    assert len(final["findings"]) == 3  # deduped across sub-questions
    assert final["report"].startswith("# Research Report: What is LangGraph?")
    assert "## Sources" in final["report"]
    assert "[1]" in final["report"] and "[3]" in final["report"]
    assert final["error"] is None


def test_run_research_empty_query_raises(mock_llm, mock_search):
    with pytest.raises(ValueError, match="query must not be empty"):
        run_research("   ", mock_llm, mock_search)


def test_run_research_with_no_results_still_terminates(mock_llm):
    """Empty search must retry once, then produce an honest empty report."""
    final = run_research(
        "What is LangGraph?", mock_llm, EmptySearchClient(), max_retries=1
    )
    assert final["findings"] == []
    assert "No sources were found" in final["report"]
    assert "No sources." in final["report"]
    # Two research rounds: initial + one retry, then the writer.
    assert final["attempts"] == 2


def test_run_research_is_deterministic(mock_llm, mock_search):
    first = run_research("What is LangGraph?", mock_llm, mock_search)
    second = run_research("What is LangGraph?", MockLLM(), MockSearchClient())
    assert first["report"] == second["report"]


def test_build_graph_returns_compiled_graph(mock_llm, mock_search):
    graph = build_graph(mock_llm, mock_search)
    assert graph is not None
    # A compiled graph exposes invoke.
    assert callable(getattr(graph, "invoke", None))
