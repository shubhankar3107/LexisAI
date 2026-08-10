from __future__ import annotations

from app.services.llm_provider import LLMProvider
from app.services.schemas.llm import (
    LLMRequest,
    LLMResponse,
)


class LLMService:
    """Application service for LLM generation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
    ):
        self._llm_provider = llm_provider

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        return self._llm_provider.generate(
            request,
        )