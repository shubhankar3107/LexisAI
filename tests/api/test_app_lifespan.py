from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.ollama_llm_provider import (
    OllamaLLMProvider,
)
from app.services.schemas.llm_config import (
    LLMProviderConfig,
)


def test_app_lifespan_initializes_and_closes_llm_provider():
    provider = MagicMock(
        spec=OllamaLLMProvider,
    )

    config = LLMProviderConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="qwen3:8b",
        timeout=60.0,
    )

    with patch(
        "app.main.OllamaLLMProvider",
        return_value=provider,
    ) as mock_provider, patch(
        "app.main.get_llm_provider_config",
        return_value=config,
    ) as mock_config:

        with TestClient(app):

            assert (
                app.state.llm_provider_registry
                is not None
            )

            resolved_provider = (
                app.state.llm_provider_registry.get(
                    "ollama",
                )
            )

            assert resolved_provider is provider

        provider.close.assert_called_once()

        mock_provider.assert_called_once_with(
            config,
        )

        mock_config.assert_called_once()