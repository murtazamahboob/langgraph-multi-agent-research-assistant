"""LLM clients: a deterministic offline mock and an optional real model.

The pipeline runs fully offline by default:

- :class:`MockLLM` — deterministic, no network, no API key. It turns a
  research question into a plan and writes an executive summary from the
  collected findings. The demo, the CLI default and the whole test suite
  use it.
- :class:`OpenAILLM` — wraps ``langchain_openai.ChatOpenAI``. It is only
  constructed when ``OPENAI_API_KEY`` is set in the environment (or passed
  explicitly). Use it with ``--live`` for real-model output.

Both clients implement the same two-method interface so the agents never
know which one they are talking to.
"""

from __future__ import annotations

import os
import re
from typing import Protocol

from .search import SearchResult


class LLMClient(Protocol):
    """Minimal interface every LLM backend must implement."""

    def plan(self, query: str) -> list[str]:
        """Turn a research question into 1-3 focused sub-questions."""

    def summarize(self, query: str, findings: list[SearchResult]) -> str:
        """Write a 2-4 sentence executive summary from the findings."""


class MockLLM:
    """Deterministic offline LLM. Same input always gives the same output."""

    def plan(self, query: str) -> list[str]:
        query = query.strip()
        if not query:
            return []
        return [
            f"Definition and core concepts of {query}",
            f"How {query} works in practice",
            f"Latest developments and use cases for {query}",
        ]

    def summarize(self, query: str, findings: list[SearchResult]) -> str:
        if not findings:
            return (
                f"No reliable sources were found for '{query}', so this "
                "report states what was attempted and nothing more. Try "
                "rephrasing the question or running with live search."
            )
        sources = "source" if len(findings) == 1 else "sources"
        return (
            f"This report answers '{query}' using {len(findings)} {sources}. "
            "The findings below cover the definition and core concepts, "
            "how the topic works in practice, and the latest developments. "
            "Every claim is tied to a numbered citation in the sources list."
        )


class OpenAILLM:
    """Real LLM via OpenAI. Requires ``OPENAI_API_KEY`` to be set."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment or "
                "run without --live to use the offline mock LLM."
            )
        from langchain_openai import ChatOpenAI

        self._llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def _complete(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        response = self._llm.invoke([HumanMessage(content=prompt)])
        content = response.content
        return content if isinstance(content, str) else str(content)

    def plan(self, query: str) -> list[str]:
        prompt = (
            "You are a research planner. Break this research question into "
            "exactly 3 focused sub-questions, one per line, numbered 1-3. "
            "No preamble, no extra text.\n\n"
            f"Research question: {query}"
        )
        lines = self._complete(prompt).splitlines()
        questions = []
        for line in lines:
            cleaned = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
            if cleaned:
                questions.append(cleaned)
        return questions[:3] or [query]

    def summarize(self, query: str, findings: list[SearchResult]) -> str:
        if not findings:
            return MockLLM().summarize(query, findings)
        evidence = "\n".join(
            f"- {finding.title}: {finding.snippet}" for finding in findings[:8]
        )
        prompt = (
            "You are a research writer. Write a 2-4 sentence executive "
            "summary that answers the research question using ONLY the "
            "evidence below. Do not invent facts.\n\n"
            f"Research question: {query}\n\nEvidence:\n{evidence}"
        )
        return self._complete(prompt).strip()


def get_llm(use_mock: bool = True, model: str = "gpt-4o-mini") -> LLMClient:
    """Factory: mock by default, real OpenAI model when ``use_mock`` is False."""
    if use_mock:
        return MockLLM()
    return OpenAILLM(model=model)
