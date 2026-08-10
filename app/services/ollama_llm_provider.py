from __future__ import annotations

import httpx

from app.services.llm_provider import LLMProvider
from app.services.schemas.llm import (
    LLMRequest,
    LLMResponse,
)
from app.services.schemas.llm_config import (
    LLMProviderConfig,
)


class LLMProviderError(Exception):
    """Base exception for LLM provider failures."""


class LLMProviderConnectionError(
    LLMProviderError,
):
    """Raised when the LLM provider cannot be reached."""


class LLMProviderTimeoutError(
    LLMProviderError,
):
    """Raised when the LLM provider request times out."""


class LLMProviderResponseError(
    LLMProviderError,
):
    """Raised when the LLM provider returns an invalid response."""


class OllamaLLMProvider(LLMProvider):
    """
    LLMProvider implementation backed by Ollama.

    Ollama-specific HTTP behavior is isolated inside this adapter.
    The rest of the application interacts only with LLMProvider.
    """

    CHAT_PATH = "/api/chat"

    def __init__(
        self,
        config: LLMProviderConfig,
        client: httpx.Client | None = None,
    ):
        if config.base_url is None:
            raise ValueError(
                "Ollama provider requires a base URL",
            )

        if config.model is None:
            raise ValueError(
                "Ollama provider requires a model",
            )

        self._config = config
        self._owns_client = client is None

        self._client = client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout,
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        payload = {
            "model": self._config.model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in request.messages
            ],
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }

        if request.max_tokens is not None:
            payload["options"]["num_predict"] = (
                request.max_tokens
            )

        try:
            response = self._client.post(
                self.CHAT_PATH,
                json=payload,
            )
            response.raise_for_status()

        except httpx.TimeoutException as exc:
            raise LLMProviderTimeoutError(
                "Ollama request timed out",
            ) from exc

        except httpx.ConnectError as exc:
            raise LLMProviderConnectionError(
                "Unable to connect to Ollama",
            ) from exc

        except httpx.HTTPError as exc:
            raise LLMProviderError(
                "Ollama request failed",
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMProviderResponseError(
                "Ollama returned invalid JSON",
            ) from exc

        content = data.get("message", {}).get(
            "content",
        )

        if not isinstance(content, str):
            raise LLMProviderResponseError(
                "Ollama response is missing message content",
            )

        model = data.get("model")

        if model is not None and not isinstance(
            model,
            str,
        ):
            raise LLMProviderResponseError(
                "Ollama response contains an invalid model",
            )

        usage = self._extract_usage(
            data,
        )

        return LLMResponse(
            content=content,
            model=model or self._config.model,
            usage=usage,
        )

    @staticmethod
    def _extract_usage(
        data: dict,
    ) -> dict[str, int]:
        usage: dict[str, int] = {}

        if isinstance(
            data.get("prompt_eval_count"),
            int,
        ):
            usage["input_tokens"] = (
                data["prompt_eval_count"]
            )

        if isinstance(
            data.get("eval_count"),
            int,
        ):
            usage["output_tokens"] = (
                data["eval_count"]
            )

        return usage

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaLLMProvider":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()