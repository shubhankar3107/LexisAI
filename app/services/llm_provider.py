from abc import ABC, abstractmethod

from app.services.schemas.llm import (
    LLMRequest,
    LLMResponse,
)


class LLMProvider(ABC):

    @abstractmethod
    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        """Generate a response for an LLM request."""
        raise NotImplementedError