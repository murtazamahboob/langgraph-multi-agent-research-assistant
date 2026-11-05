"""Tests for the CLI entry point."""

from __future__ import annotations

from research_assistant.cli import main, slugify


def test_slugify():
    assert slugify("What is LangGraph?") == "what-is-langgraph"
    assert slugify("  AI & ML: 2026! ") == "ai-ml-2026"
    assert slugify("!!!") == "report"


def test_cli_writes_report_to_default_location(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["What is LangGraph?"]) == 0
    report_path = tmp_path / "outputs" / "what-is-langgraph.md"
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "# Research Report: What is LangGraph?" in text
    assert "## Sources" in text
    out = capsys.readouterr().out
    assert "Report:" in out
    assert "Sources: 3 finding(s)" in out


def test_cli_respects_custom_output_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    custom = tmp_path / "custom" / "my-report.md"
    assert main(["What is LangGraph?", "--output", str(custom)]) == 0
    assert custom.exists()
    assert "What is LangGraph?" in custom.read_text(encoding="utf-8")


def test_cli_empty_query_returns_error_code(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["   "]) == 2
    assert "Error" in capsys.readouterr().err
