"""LangGraph Multi-Agent Research Assistant.

A supervisor-style multi-agent pipeline built with LangChain + LangGraph:
a planner agent breaks a question into sub-questions, a researcher agent
searches the web for each one, and a writer agent produces a cited
Markdown report. Runs fully offline with deterministic mock backends,
or live with DuckDuckGo search and an OpenAI model.
"""

from __future__ import annotations

from .config import Settings
from .graph import build_graph, run_research
from .llm import MockLLM, OpenAILLM, get_llm
from .search import DuckDuckGoSearchClient, MockSearchClient, SearchResult

__all__ = [
    "Settings",
    "build_graph",
    "run_research",
    "MockLLM",
    "OpenAILLM",
    "get_llm",
    "DuckDuckGoSearchClient",
    "MockSearchClient",
    "SearchResult",
]

__version__ = "1.0.0"
