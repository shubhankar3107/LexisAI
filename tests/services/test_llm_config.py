import pytest

from app.services.schemas.llm_config import (
    LLMProviderConfig,
)


def test_llm_provider_config_accepts_provider_only():
    config = LLMProviderConfig(
        provider="test-provider",
    )

    assert config.provider == "test-provider"
    assert config.base_url is None
    assert config.api_key is None
    assert config.model is None
    assert config.timeout == 60.0


def test_llm_provider_config_accepts_full_configuration():
    config = LLMProviderConfig(
        provider="test-provider",
        base_url="http://localhost:8000/v1",
        api_key="test-api-key",
        model="test-model",
        timeout=30.0,
    )

    assert config.provider == "test-provider"
    assert config.base_url == (
        "http://localhost:8000/v1"
    )
    assert config.api_key == "test-api-key"
    assert config.model == "test-model"
    assert config.timeout == 30.0


def test_llm_provider_config_is_immutable():
    config = LLMProviderConfig(
        provider="test-provider",
    )

    with pytest.raises(
        AttributeError,
    ):
        config.provider = "another-provider"


def test_llm_provider_config_rejects_empty_provider():
    with pytest.raises(
        ValueError,
        match="LLM provider must not be empty",
    ):
        LLMProviderConfig(
            provider="",
        )


def test_llm_provider_config_rejects_whitespace_provider():
    with pytest.raises(
        ValueError,
        match="LLM provider must not be empty",
    ):
        LLMProviderConfig(
            provider="   ",
        )


def test_llm_provider_config_rejects_empty_base_url():
    with pytest.raises(
        ValueError,
        match="LLM base URL must not be empty",
    ):
        LLMProviderConfig(
            provider="test-provider",
            base_url="   ",
        )


def test_llm_provider_config_rejects_empty_model():
    with pytest.raises(
        ValueError,
        match="LLM model must not be empty",
    ):
        LLMProviderConfig(
            provider="test-provider",
            model="",
        )


def test_llm_provider_config_rejects_non_positive_timeout():
    with pytest.raises(
        ValueError,
        match="LLM timeout must be greater than zero",
    ):
        LLMProviderConfig(
            provider="test-provider",
            timeout=0,
        )


def test_llm_provider_config_rejects_negative_timeout():
    with pytest.raises(
        ValueError,
        match="LLM timeout must be greater than zero",
    ):
        LLMProviderConfig(
            provider="test-provider",
            timeout=-1,
        )