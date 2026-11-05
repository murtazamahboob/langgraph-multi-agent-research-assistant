"""Agent nodes for the research graph.

Three agents, one responsibility each:

- **Planner** — turns the research question into focused sub-questions.
- **Researcher** — searches the web for every sub-question, dedupes by URL.
- **Writer** — writes the executive summary and renders the final report.

A supervisor edge (see :mod:`research_assistant.graph`) retries the
planner once with the raw question when the first search round returns
nothing. The writer always runs last and never invents sources.
"""

from __future__ import annotations

from .llm import LLMClient
from .report import findings_to_search_results, render_report
from .search import SearchClient
from .state import Finding, ResearchState


def dedupe_findings(findings: list[Finding]) -> list[Finding]:
    """Remove duplicate URLs, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[Finding] = []
    for finding in findings:
        if finding["url"] not in seen:
            seen.add(finding["url"])
            unique.append(finding)
    return unique


def make_planner_node(llm: LLMClient):
    """Return the planner node bound to ``llm``."""

    def planner_node(state: ResearchState) -> dict:
        if state["attempts"] > 0:
            # Retry round: search the raw question directly.
            return {"plan": [state["query"]]}
        plan = llm.plan(state["query"])
        return {"plan": plan or [state["query"]]}

    return planner_node


def make_researcher_node(search_client: SearchClient, max_results: int = 3):
    """Return the researcher node bound to ``search_client``."""

    def researcher_node(state: ResearchState) -> dict:
        collected: list[Finding] = list(state.get("findings") or [])
        for sub_question in state.get("plan") or [state["query"]]:
            for result in search_client.search(sub_question, max_results=max_results):
                collected.append(
                    Finding(
                        sub_question=sub_question,
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                    )
                )
        return {"findings": dedupe_findings(collected)}

    return researcher_node


def make_writer_node(llm: LLMClient):
    """Return the writer node bound to ``llm``."""

    def writer_node(state: ResearchState) -> dict:
        findings = state.get("findings") or []
        summary = llm.summarize(state["query"], findings_to_search_results(findings))
        report = render_report(state["query"], summary, findings)
        return {"report": report}

    return writer_node
