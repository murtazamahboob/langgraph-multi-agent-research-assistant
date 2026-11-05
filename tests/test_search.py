"""Tests for the search clients and dedup helpers."""

from __future__ import annotations

import pytest

from research_assistant.search import (
    DuckDuckGoSearchClient,
    MockSearchClient,
    SearchResult,
    dedupe_results,
)


def test_mock_search_returns_results_for_query():
    results = MockSearchClient().search("What is LangGraph?", max_results=3)
    assert len(results) == 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.title and r.url and r.snippet for r in results)


def test_mock_search_is_deterministic():
    client = MockSearchClient()
    first = client.search("LangGraph")
    second = client.search("LangGraph")
    assert first == second


def test_mock_search_empty_query_returns_empty_list():
    assert MockSearchClient().search("   ") == []


def test_mock_search_respects_max_results():
    assert len(MockSearchClient().search("LangGraph", max_results=2)) == 2
    assert len(MockSearchClient().search("LangGraph", max_results=1)) == 1


def test_mock_search_unwraps_planner_sub_questions():
    """Canned titles must read naturally, not repeat the plan wrapper."""
    results = MockSearchClient().search(
        "Definition and core concepts of What is LangGraph?", max_results=3
    )
    assert results[0].title == "What is LangGraph? — overview and core concepts"
    assert results[1].title == "How What is LangGraph? works in practice"
    assert results[2].title == (
        "What is LangGraph? — latest developments and use cases"
    )
    assert "Definition and core concepts of What" not in results[0].title


def test_dedupe_results_removes_duplicate_urls():
    results = [
        SearchResult(title="A", url="https://example.com/a", snippet="s1"),
        SearchResult(title="A again", url="https://example.com/a", snippet="s2"),
        SearchResult(title="B", url="https://example.com/b", snippet="s3"),
    ]
    deduped = dedupe_results(results)
    assert [r.url for r in deduped] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert deduped[0].title == "A"  # first occurrence kept


def test_live_search_empty_query_returns_empty_without_network():
    assert DuckDuckGoSearchClient().search("   ") == []


def test_live_search_failure_returns_empty_list(monkeypatch):
    """A search backend failure must never crash the pipeline."""

    class ExplodingDDGS:
        def text(self, *args, **kwargs):
            raise RuntimeError("no network")

    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", ExplodingDDGS)
    assert DuckDuckGoSearchClient().search("LangGraph") == []
