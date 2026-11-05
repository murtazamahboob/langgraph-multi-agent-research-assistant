"""Tests for the LLM clients (mock + real-model factory)."""

from __future__ import annotations

import pytest

from research_assistant.llm import MockLLM, OpenAILLM, get_llm
from research_assistant.search import SearchResult


def test_mock_plan_returns_three_sub_questions():
    plan = MockLLM().plan("What is LangGraph?")
    assert len(plan) == 3
    assert all("What is LangGraph?" in item for item in plan)
    assert len(set(plan)) == 3  # all distinct


def test_mock_plan_is_deterministic():
    llm = MockLLM()
    assert llm.plan("LangGraph agents") == llm.plan("LangGraph agents")


def test_mock_plan_empty_query_returns_empty_list():
    assert MockLLM().plan("   ") == []


def test_mock_summarize_counts_sources():
    findings = [
        SearchResult(title="A", url="https://example.com/a", snippet="s1"),
        SearchResult(title="B", url="https://example.com/b", snippet="s2"),
    ]
    summary = MockLLM().summarize("What is LangGraph?", findings)
    assert "2 sources" in summary
    assert "What is LangGraph?" in summary


def test_mock_summarize_no_findings_is_honest():
    summary = MockLLM().summarize("What is LangGraph?", [])
    assert "No reliable sources" in summary
    assert "rephrasing" in summary


def test_get_llm_mock_by_default():
    assert isinstance(get_llm(), MockLLM)
    assert isinstance(get_llm(use_mock=True), MockLLM)


def test_openai_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        OpenAILLM()


def test_get_llm_live_without_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        get_llm(use_mock=False)
