"""Markdown report rendering with numbered citations.

The writer agent produces the executive summary; this module turns the
summary plus the collected findings into a complete, self-contained
Markdown report. Citations are numbered ``[1]``, ``[2]``, ... in the
order findings were collected, and every citation resolves to a URL in
the Sources section. A report with no findings says so explicitly
instead of inventing content.
"""

from __future__ import annotations

from datetime import date

from .search import SearchResult
from .state import Finding


def render_report(
    query: str,
    summary: str,
    findings: list[Finding],
    *,
    generated_by: str = "LangGraph Multi-Agent Research Assistant",
) -> str:
    """Render the final Markdown report for ``query``."""
    lines: list[str] = [f"# Research Report: {query}", ""]
    lines.append(f"_Generated on {date.today().isoformat()} by {generated_by}._")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary.strip())
    lines.append("")

    if not findings:
        lines.append("## Findings")
        lines.append("")
        lines.append(
            "No sources were found for this question. The search ran, "
            "returned no usable results, and no claims are made below."
        )
        lines.append("")
    else:
        lines.append("## Key Findings")
        lines.append("")
        for index, finding in enumerate(findings, start=1):
            lines.append(f"### {finding['title']} [{index}]")
            lines.append("")
            lines.append(finding["snippet"].strip())
            lines.append("")
            lines.append(f"_Source: {finding['url']}_")
            lines.append("")

    lines.append("## Sources")
    lines.append("")
    if not findings:
        lines.append("No sources.")
    else:
        for index, finding in enumerate(findings, start=1):
            lines.append(f"[{index}] {finding['title']} — {finding['url']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "This report was produced by an automated multi-agent pipeline "
        "(planner → researcher → writer). Verify important claims against "
        "the linked sources before citing them elsewhere."
    )
    lines.append("")
    return "\n".join(lines)


def findings_to_search_results(findings: list[Finding]) -> list[SearchResult]:
    """Convert stored state findings back to :class:`SearchResult` objects."""
    return [
        SearchResult(
            title=finding["title"], url=finding["url"], snippet=finding["snippet"]
        )
        for finding in findings
    ]
