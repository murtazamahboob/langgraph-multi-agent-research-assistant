"""Multi-agent research assistant state definition.

The LangGraph state is a plain ``TypedDict``. Every node returns a partial
update; LangGraph merges updates into the running state. Lists are always
replaced wholesale by the node that owns them (no append reducers), which
keeps the flow deterministic and easy to test.
"""

from __future__ import annotations

from typing import TypedDict


class Finding(TypedDict):
    """One search result, kept in the graph state."""

    sub_question: str
    title: str
    url: str
    snippet: str


class ResearchState(TypedDict):
    """State carried through the supervisor -> researcher -> writer graph."""

    query: str
    plan: list[str]
    findings: list[Finding]
    report: str
    attempts: int
    error: str | None
