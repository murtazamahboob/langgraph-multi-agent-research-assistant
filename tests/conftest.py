"""Shared fixtures for the research-assistant test suite."""

from __future__ import annotations

import pytest

from research_assistant.llm import MockLLM
from research_assistant.search import MockSearchClient
from research_assistant.state import ResearchState


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture()
def mock_search() -> MockSearchClient:
    return MockSearchClient()


@pytest.fixture()
def fresh_state() -> ResearchState:
    return {
        "query": "What is LangGraph?",
        "plan": [],
        "findings": [],
        "report": "",
        "attempts": 0,
        "error": None,
    }
