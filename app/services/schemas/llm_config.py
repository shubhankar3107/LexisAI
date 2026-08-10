from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProviderConfig:
    """
    Provider-neutral configuration for an LLM backend.

    This object contains infrastructure configuration only.
    It does not define provider-specific behavior.
    """

    provider: str
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError(
                "LLM provider must not be empty",
            )

        if self.base_url is not None:
            if not self.base_url.strip():
                raise ValueError(
                    "LLM base URL must not be empty",
                )

        if self.model is not None:
            if not self.model.strip():
                raise ValueError(
                    "LLM model must not be empty",
                )

        if self.timeout <= 0:
            raise ValueError(
                "LLM timeout must be greater than zero",
            )