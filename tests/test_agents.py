"""Tests for the planner / researcher / writer agent nodes."""

from __future__ import annotations

from research_assistant.agents import (
    dedupe_findings,
    make_planner_node,
    make_researcher_node,
    make_writer_node,
)
from research_assistant.state import Finding


def test_planner_node_builds_three_step_plan(mock_llm, fresh_state):
    update = make_planner_node(mock_llm)(fresh_state)
    assert len(update["plan"]) == 3
    assert all(isinstance(item, str) and item for item in update["plan"])


def test_planner_node_retry_uses_raw_query(mock_llm, fresh_state):
    fresh_state["attempts"] = 1
    update = make_planner_node(mock_llm)(fresh_state)
    assert update["plan"] == ["What is LangGraph?"]


def test_researcher_node_collects_and_dedupes(mock_search, fresh_state):
    fresh_state["plan"] = ["q1", "q2", "q3"]
    update = make_researcher_node(mock_search, max_results=3)(fresh_state)
    # The mock returns the same 3 URLs for every sub-question: 9 raw hits
    # must collapse to 3 unique findings.
    assert len(update["findings"]) == 3
    urls = [f["url"] for f in update["findings"]]
    assert len(set(urls)) == 3
    assert all(isinstance(f, dict) and f["title"] and f["url"] for f in update["findings"])


def test_researcher_node_keeps_previous_findings(mock_search, fresh_state):
    fresh_state["plan"] = ["q1"]
    fresh_state["findings"] = [
        Finding(
            sub_question="old",
            title="Old",
            url="https://example.com/old",
            snippet="old snippet",
        )
    ]
    update = make_researcher_node(mock_search, max_results=1)(fresh_state)
    urls = [f["url"] for f in update["findings"]]
    assert "https://example.com/old" in urls
    assert len(update["findings"]) == 2


def test_writer_node_produces_cited_report(mock_llm, fresh_state):
    fresh_state["findings"] = [
        Finding(
            sub_question="q1",
            title="LangGraph overview",
            url="https://example.com/research/overview",
            snippet="LangGraph builds stateful agents.",
        )
    ]
    update = make_writer_node(mock_llm)(fresh_state)
    report = update["report"]
    assert "What is LangGraph?" in report
    assert "[1]" in report
    assert "https://example.com/research/overview" in report
    assert "## Sources" in report


def test_writer_node_with_no_findings_is_honest(mock_llm, fresh_state):
    update = make_writer_node(mock_llm)(fresh_state)
    assert "No sources were found" in update["report"]
    assert "No sources." in update["report"]


def test_dedupe_findings_keeps_first_occurrence():
    findings = [
        Finding(sub_question="a", title="A", url="https://x/a", snippet="1"),
        Finding(sub_question="b", title="A2", url="https://x/a", snippet="2"),
        Finding(sub_question="c", title="B", url="https://x/b", snippet="3"),
    ]
    deduped = dedupe_findings(findings)
    assert [f["url"] for f in deduped] == ["https://x/a", "https://x/b"]
    assert deduped[0]["title"] == "A"
