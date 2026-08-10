import httpx
import pytest

from app.services.llm_provider import LLMProvider
from app.services.ollama_llm_provider import (
    LLMProviderConnectionError,
    LLMProviderError,
    LLMProviderResponseError,
    LLMProviderTimeoutError,
    OllamaLLMProvider,
)
from app.services.schemas.llm import (
    LLMMessage,
    LLMRequest,
)
from app.services.schemas.llm_config import (
    LLMProviderConfig,
)


def make_config():
    return LLMProviderConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="qwen3:8b",
        timeout=30.0,
    )


def make_request():
    return LLMRequest(
        messages=[
            LLMMessage(
                role="system",
                content="You are a legal assistant.",
            ),
            LLMMessage(
                role="user",
                content="What is the notice period?",
            ),
        ],
        temperature=0.2,
        max_tokens=100,
    )


def make_response(
    payload,
    status_code=200,
):
    request = httpx.Request(
        "POST",
        "http://localhost:11434/api/chat",
    )

    return httpx.Response(
        status_code=status_code,
        json=payload,
        request=request,
    )


class FakeClient:

    def __init__(
        self,
        response,
    ):
        self.response = response
        self.received_url = None
        self.received_json = None

    def post(
        self,
        url,
        json,
    ):
        self.received_url = url
        self.received_json = json

        return self.response


def test_ollama_provider_implements_llm_provider():
    client = FakeClient(
        make_response(
            {
                "model": "qwen3:8b",
                "message": {
                    "content": "30 days notice.",
                },
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    assert isinstance(
        provider,
        LLMProvider,
    )


def test_provider_sends_chat_request():
    client = FakeClient(
        make_response(
            {
                "model": "qwen3:8b",
                "message": {
                    "content": "30 days notice.",
                },
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    response = provider.generate(
        make_request(),
    )

    assert response.content == (
        "30 days notice."
    )

    assert response.model == "qwen3:8b"

    assert client.received_url == (
        "/api/chat"
    )

    assert client.received_json["model"] == (
        "qwen3:8b"
    )

    assert client.received_json["stream"] is False

    assert client.received_json["options"] == {
        "temperature": 0.2,
        "num_predict": 100,
    }

    assert client.received_json["messages"] == [
        {
            "role": "system",
            "content": "You are a legal assistant.",
        },
        {
            "role": "user",
            "content": "What is the notice period?",
        },
    ]


def test_provider_extracts_usage():
    client = FakeClient(
        make_response(
            {
                "model": "qwen3:8b",
                "message": {
                    "content": "30 days notice.",
                },
                "prompt_eval_count": 42,
                "eval_count": 17,
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    response = provider.generate(
        make_request(),
    )

    assert response.usage == {
        "input_tokens": 42,
        "output_tokens": 17,
    }


def test_provider_handles_missing_usage():
    client = FakeClient(
        make_response(
            {
                "model": "qwen3:8b",
                "message": {
                    "content": "30 days notice.",
                },
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    response = provider.generate(
        make_request(),
    )

    assert response.usage == {}


def test_provider_rejects_missing_base_url():
    config = LLMProviderConfig(
        provider="ollama",
        model="qwen3:8b",
    )

    with pytest.raises(
        ValueError,
        match="Ollama provider requires a base URL",
    ):
        OllamaLLMProvider(
            config=config,
        )


def test_provider_rejects_missing_model():
    config = LLMProviderConfig(
        provider="ollama",
        base_url="http://localhost:11434",
    )

    with pytest.raises(
        ValueError,
        match="Ollama provider requires a model",
    ):
        OllamaLLMProvider(
            config=config,
        )


def test_provider_translates_http_status_error():
    client = FakeClient(
        make_response(
            {
                "error": "model not found",
            },
            status_code=404,
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderError,
        match="Ollama request failed",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_translates_timeout_error():
    client = FakeClient(
        None,
    )

    def failing_post(
        url,
        json,
    ):
        raise httpx.ReadTimeout(
            "timeout",
        )

    client.post = failing_post

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderTimeoutError,
        match="Ollama request timed out",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_translates_connection_error():
    client = FakeClient(
        None,
    )

    def failing_post(
        url,
        json,
    ):
        raise httpx.ConnectError(
            "connection refused",
        )

    client.post = failing_post

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderConnectionError,
        match="Unable to connect to Ollama",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_raises_http_error_for_unexpected_status():
    client = FakeClient(
        make_response(
            {
                "error": "model not found",
            },
            status_code=404,
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderError,
        match="Ollama request failed",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_rejects_invalid_json():
    request = httpx.Request(
        "POST",
        "http://localhost:11434/api/chat",
    )

    response = httpx.Response(
        status_code=200,
        content=b"not json",
        request=request,
    )

    client = FakeClient(
        response,
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderResponseError,
        match="Ollama returned invalid JSON",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_rejects_missing_message_content():
    client = FakeClient(
        make_response(
            {
                "model": "qwen3:8b",
                "message": {},
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderResponseError,
        match="missing message content",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_does_not_close_injected_client():
    class RecordingClient:

        def __init__(self):
            self.closed = False

        def post(self, url, json):
            return make_response(
                {
                    "model": "qwen3:8b",
                    "message": {
                        "content": "30 days notice.",
                    },
                },
            )

        def close(self):
            self.closed = True

    client = RecordingClient()

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    provider.close()

    assert client.closed is False


def test_provider_closes_owned_client():
    provider = OllamaLLMProvider(
        config=make_config(),
    )

    client = provider._client

    provider.close()

    # httpx.Client exposes this state after close().
    assert client.is_closed is True


def test_provider_context_manager_closes_owned_client():
    provider = OllamaLLMProvider(
        config=make_config(),
    )

    client = provider._client

    with provider:
        assert client.is_closed is False

    assert client.is_closed is True


def test_provider_rejects_invalid_response_model():
    client = FakeClient(
        make_response(
            {
                "model": 123,
                "message": {
                    "content": "30 days notice.",
                },
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    with pytest.raises(
        LLMProviderResponseError,
        match="Ollama response contains an invalid model",
    ):
        provider.generate(
            make_request(),
        )


def test_provider_ignores_invalid_usage_values():
    client = FakeClient(
        make_response(
            {
                "model": "qwen3:8b",
                "message": {
                    "content": "30 days notice.",
                },
                "prompt_eval_count": "42",
                "eval_count": None,
            },
        ),
    )

    provider = OllamaLLMProvider(
        config=make_config(),
        client=client,
    )

    response = provider.generate(
        make_request(),
    )

    assert response.usage == {}