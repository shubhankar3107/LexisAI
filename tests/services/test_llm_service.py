from app.services.llm_provider import LLMProvider
from app.services.llm_service import LLMService
from app.services.schemas.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
)


class FakeLLMProvider(LLMProvider):

    def __init__(self):
        self.received_request = None

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.received_request = request

        return LLMResponse(
            content="The contract requires 30 days notice.",
            model="fake-model",
            usage={
                "input_tokens": 20,
                "output_tokens": 10,
            },
        )


def test_llm_service_delegates_to_provider():
    provider = FakeLLMProvider()
    service = LLMService(provider)

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="What is the termination notice?",
            ),
        ],
    )

    response = service.generate(request)

    assert response.content == (
        "The contract requires 30 days notice."
    )

    assert response.model == "fake-model"

    assert response.usage == {
        "input_tokens": 20,
        "output_tokens": 10,
    }

    assert provider.received_request is request


import pytest
from unittest.mock import Mock

from app.services.llm_service import LLMService
from app.services.ollama_llm_provider import (
    LLMProviderConnectionError,
)
from app.services.schemas.llm import (
    LLMMessage,
    LLMRequest,
)


def test_generate_propagates_provider_failure():
    provider = Mock()

    provider.generate.side_effect = (
        LLMProviderConnectionError(
            "Unable to connect to Ollama",
        )
    )

    service = LLMService(
        llm_provider=provider,
    )

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Hello",
            ),
        ],
    )

    with pytest.raises(
        LLMProviderConnectionError,
        match="Unable to connect to Ollama",
    ):
        service.generate(request)

    provider.generate.assert_called_once_with(
        request,
    )