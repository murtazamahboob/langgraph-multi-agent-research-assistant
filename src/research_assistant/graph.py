"""LangGraph wiring: supervisor -> researcher -> writer.

Graph shape::

    START -> planner -> researcher -> router
    router: findings found  -> writer -> END
    router: no findings, retries left -> planner (attempts += 1)
    router: no findings, retries used up -> writer (reports "no sources")

The researcher node counts its own rounds in ``state["attempts"]``, so
the retry always terminates. The graph is built from injected ``llm``
and ``search_client`` objects, so the same graph runs offline with the
mock backends and live with real search + a real LLM.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import make_planner_node, make_researcher_node, make_writer_node
from .llm import LLMClient
from .search import SearchClient
from .state import ResearchState


def build_graph(
    llm: LLMClient,
    search_client: SearchClient,
    *,
    max_results: int = 3,
    max_retries: int = 1,
):
    """Build and compile the multi-agent research graph."""
    graph = StateGraph(ResearchState)

    graph.add_node("planner", make_planner_node(llm))
    graph.add_node("researcher", _counting_researcher(search_client, max_results))
    graph.add_node("writer", make_writer_node(llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")

    def route_after_research(state: ResearchState) -> str:
        if state.get("findings"):
            return "write"
        if state.get("attempts", 0) <= max_retries:
            return "retry"
        return "write"

    graph.add_conditional_edges(
        "researcher",
        route_after_research,
        {"retry": "planner", "write": "writer"},
    )
    graph.add_edge("writer", END)
    return graph.compile()


def _counting_researcher(search_client: SearchClient, max_results: int):
    """Researcher node that also counts completed research rounds."""
    inner = make_researcher_node(search_client, max_results)

    def researcher_node(state: ResearchState) -> dict:
        update = inner(state)
        update["attempts"] = state.get("attempts", 0) + 1
        return update

    return researcher_node


def run_research(
    query: str,
    llm: LLMClient,
    search_client: SearchClient,
    *,
    max_results: int = 3,
    max_retries: int = 1,
) -> ResearchState:
    """Run the graph for ``query`` and return the final state.

    Raises:
        ValueError: if ``query`` is empty.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    graph = build_graph(
        llm, search_client, max_results=max_results, max_retries=max_retries
    )
    initial_state: ResearchState = {
        "query": query.strip(),
        "plan": [],
        "findings": [],
        "report": "",
        "attempts": 0,
        "error": None,
    }
    result = graph.invoke(initial_state)
    return result  # type: ignore[return-value]
