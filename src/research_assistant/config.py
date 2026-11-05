"""Project settings, read from the environment with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the research assistant."""

    model: str = "gpt-4o-mini"
    max_results_per_query: int = 3
    max_retries: int = 1
    output_dir: str = "outputs"

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables (all optional)."""
        return cls(
            model=os.environ.get("RA_MODEL", "gpt-4o-mini"),
            max_results_per_query=int(
                os.environ.get("RA_MAX_RESULTS", "3")
            ),
            max_retries=int(os.environ.get("RA_MAX_RETRIES", "1")),
            output_dir=os.environ.get("RA_OUTPUT_DIR", "outputs"),
        )
