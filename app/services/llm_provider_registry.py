from __future__ import annotations

from app.services.llm_provider import LLMProvider

from app.services.exceptions import (
    LLMProviderNotFoundError,
)


class LLMProviderRegistry:
    """
    Registry for resolving configured LLM providers.

    The registry knows only provider identifiers and provider
    instances. It contains no provider-specific generation logic.
    """

    def __init__(
        self,
        providers: dict[str, LLMProvider],
    ):
        self._providers = providers.copy()

    def get(
        self,
        name: str,
    ) -> LLMProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise LLMProviderNotFoundError(
                f"LLM provider '{name}' not found",
            ) from exc