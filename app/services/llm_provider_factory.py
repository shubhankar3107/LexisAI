from __future__ import annotations

from collections.abc import Callable

from app.services.llm_provider import LLMProvider
from app.services.schemas.llm_config import LLMProviderConfig
from app.services.exceptions import (
    LLMProviderNotFoundError,
)


class LLMProviderFactory:
    """
    Construct LLM providers from provider-neutral configuration.

    Concrete provider constructors are registered externally so this
    factory does not contain provider-specific branching.
    """

    def __init__(
        self,
        constructors: dict[
            str,
            Callable[[LLMProviderConfig], LLMProvider],
        ],
    ):
        self._constructors = constructors.copy()

    def create(
        self,
        config: LLMProviderConfig,
    ) -> LLMProvider:
        try:
            constructor = self._constructors[
                config.provider
            ]
        except KeyError as exc:
            raise LLMProviderNotFoundError(
                f"LLM provider '{config.provider}' "
                "is not registered",
            ) from exc

        return constructor(config)
