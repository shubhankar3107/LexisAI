import pytest

from app.services.llm_provider import LLMProvider
from app.services.schemas.llm import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
)


class FakeLLMProvider(LLMProvider):

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        return LLMResponse(
            content="Generated answer.",
            model="fake-model",
        )


def test_llm_provider_returns_response():
    provider = FakeLLMProvider()

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="What is this contract about?",
            ),
        ],
    )

    response = provider.generate(request)

    assert isinstance(response, LLMResponse)
    assert response.content == "Generated answer."
    assert response.model == "fake-model"


def test_llm_provider_receives_request():
    class RecordingLLMProvider(LLMProvider):

        def __init__(self):
            self.received_request = None

        def generate(
            self,
            request: LLMRequest,
        ) -> LLMResponse:
            self.received_request = request

            return LLMResponse(
                content="ok",
            )

    provider = RecordingLLMProvider()

    request = LLMRequest(
        messages=[
            LLMMessage(
                role="user",
                content="Explain clause 4.",
            ),
        ],
        temperature=0.2,
        max_tokens=100,
    )

    provider.generate(request)

    assert provider.received_request is request


def test_llm_provider_is_abstract():
    with pytest.raises(TypeError):
        LLMProvider()