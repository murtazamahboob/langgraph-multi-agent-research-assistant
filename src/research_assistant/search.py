"""Web search clients.

Two implementations share one small interface:

- :class:`DuckDuckGoSearchClient` — free, no API key, via the ``ddgs``
  package (the successor of ``duckduckgo_search``).
- :class:`MockSearchClient` — deterministic canned results. Used by the
  offline demo and the test suite so the whole pipeline runs with no
  network access and no API key.

A live search that fails (no network, rate limit, ...) never crashes the
graph: it logs a warning and returns an empty result list, and the
supervisor node retries once with a broader query before giving up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchResult:
    """A single web search hit."""

    title: str
    url: str
    snippet: str


class SearchClient(Protocol):
    """Minimal interface every search backend must implement."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return up to ``max_results`` hits for ``query``."""


class DuckDuckGoSearchClient:
    """Free web search via DuckDuckGo. No API key required."""

    def __init__(self, region: str = "wt-wt", safesearch: str = "moderate") -> None:
        self._region = region
        self._safesearch = safesearch

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        try:
            from ddgs import DDGS
        except ImportError:  # pragma: no cover - legacy package fallback
            from duckduckgo_search import DDGS  # type: ignore[no-redef]

        try:
            raw = DDGS().text(
                query,
                region=self._region,
                safesearch=self._safesearch,
                max_results=max_results,
            )
        except Exception as exc:  # network error, rate limit, ...
            logger.warning("DuckDuckGo search failed for %r: %s", query, exc)
            return []
        results = []
        for item in raw or []:
            title = (item.get("title") or "").strip()
            url = (item.get("href") or "").strip()
            snippet = (item.get("body") or "").strip()
            if title and url:
                results.append(SearchResult(title=title, url=url, snippet=snippet))
        return results


class MockSearchClient:
    """Deterministic offline search used by the demo and the test suite.

    Results are derived from the query text so different queries produce
    different (but stable) result sets. No network access is performed.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        self.calls.append(query)
        topic = self._extract_topic(query)
        canned = [
            SearchResult(
                title=f"{topic} — overview and core concepts",
                url="https://example.com/research/overview",
                snippet=(
                    f"An overview of {topic}, covering its definition, "
                    "core concepts, and why it matters."
                ),
            ),
            SearchResult(
                title=f"How {topic} works in practice",
                url="https://example.com/research/how-it-works",
                snippet=(
                    f"A practical walkthrough of how {topic} is used, "
                    "with examples and common patterns."
                ),
            ),
            SearchResult(
                title=f"{topic} — latest developments and use cases",
                url="https://example.com/research/latest",
                snippet=(
                    f"Recent developments, real-world use cases, and "
                    f"what is next for {topic}."
                ),
            ),
        ]
        return canned[:max_results]

    @staticmethod
    def _extract_topic(query: str) -> str:
        """Recover the original research question from a sub-question.

        Planner sub-questions wrap the question in fixed templates
        ("Definition and core concepts of X", "How X works in practice",
        "Latest developments and use cases for X"); unwrap them so the
        canned results read naturally. Anything else is truncated to
        its first 8 words.
        """
        text = query.strip()
        wrappers = [
            ("Definition and core concepts of ", ""),
            ("How ", " works in practice"),
            ("Latest developments and use cases for ", ""),
        ]
        for prefix, suffix in wrappers:
            if text.startswith(prefix) and (not suffix or text.endswith(suffix)):
                core = text[len(prefix):]
                if suffix:
                    core = core[: -len(suffix)]
                if core.strip():
                    return core.strip()
        return " ".join(text.split()[:8])


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate URLs, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        if result.url not in seen:
            seen.add(result.url)
            unique.append(result)
    return unique
