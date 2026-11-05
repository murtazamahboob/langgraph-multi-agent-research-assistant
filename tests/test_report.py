"""Tests for report rendering."""

from __future__ import annotations

from research_assistant.report import findings_to_search_results, render_report
from research_assistant.state import Finding


def _finding(**overrides):
    base = Finding(
        sub_question="q1",
        title="LangGraph overview",
        url="https://example.com/research/overview",
        snippet="LangGraph builds stateful multi-actor applications.",
    )
    base.update(overrides)  # type: ignore[arg-type]
    return base


def test_render_report_with_findings():
    report = render_report(
        "What is LangGraph?",
        "A short summary.",
        [_finding()],
    )
    assert report.startswith("# Research Report: What is LangGraph?")
    assert "## Executive Summary" in report
    assert "A short summary." in report
    assert "### LangGraph overview [1]" in report
    assert "[1] LangGraph overview — https://example.com/research/overview" in report


def test_render_report_numbers_citations_in_order():
    findings = [
        _finding(title="First", url="https://example.com/1"),
        _finding(title="Second", url="https://example.com/2"),
    ]
    report = render_report("Q", "Summary.", findings)
    assert "[1] First — https://example.com/1" in report
    assert "[2] Second — https://example.com/2" in report
    assert "### First [1]" in report
    assert "### Second [2]" in report


def test_render_report_no_findings_states_that():
    report = render_report("Q", "Nothing found.", [])
    assert "No sources were found" in report
    assert "No sources." in report
    assert "[1]" not in report


def test_findings_to_search_results_round_trip():
    findings = [_finding(title="T", url="https://example.com/t", snippet="s")]
    results = findings_to_search_results(findings)
    assert len(results) == 1
    assert results[0].title == "T"
    assert results[0].url == "https://example.com/t"
    assert results[0].snippet == "s"
