import pytest

from app.services.exceptions import (
    LLMProviderNotFoundError,
)
from app.services.llm_provider import LLMProvider
from app.services.llm_provider_factory import (
    LLMProviderFactory,
)
from app.services.schemas.llm import (
    LLMRequest,
    LLMResponse,
)
from app.services.schemas.llm_config import (
    LLMProviderConfig,
)


class FakeLLMProvider(LLMProvider):

    def __init__(
        self,
        config: LLMProviderConfig,
    ):
        self.config = config

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        return LLMResponse(
            content="test response",
            model=self.config.model,
        )


def test_factory_constructs_registered_provider():
    factory = LLMProviderFactory(
        {
            "test-provider": FakeLLMProvider,
        },
    )

    config = LLMProviderConfig(
        provider="test-provider",
        model="test-model",
    )

    provider = factory.create(config)

    assert isinstance(
        provider,
        FakeLLMProvider,
    )

    assert provider.config is config


def test_factory_passes_configuration_to_constructor():
    received = {}

    def constructor(
        config: LLMProviderConfig,
    ) -> LLMProvider:
        received["config"] = config

        return FakeLLMProvider(config)

    factory = LLMProviderFactory(
        {
            "test-provider": constructor,
        },
    )

    config = LLMProviderConfig(
        provider="test-provider",
        base_url="http://localhost:8000",
        model="test-model",
        timeout=30.0,
    )

    factory.create(config)

    assert received["config"] is config


def test_factory_rejects_unknown_provider():
    factory = LLMProviderFactory({})

    config = LLMProviderConfig(
        provider="missing-provider",
    )

    with pytest.raises(
        LLMProviderNotFoundError,
        match=(
            "LLM provider 'missing-provider' "
            "is not registered"
        ),
    ):
        factory.create(config)


def test_factory_copies_constructor_mapping():
    factory_constructors = {
        "test-provider": FakeLLMProvider,
    }

    factory = LLMProviderFactory(
        factory_constructors,
    )

    factory_constructors.clear()

    config = LLMProviderConfig(
        provider="test-provider",
    )

    provider = factory.create(config)

    assert isinstance(
        provider,
        FakeLLMProvider,
    )


def test_factory_returns_llm_provider_contract():
    factory = LLMProviderFactory(
        {
            "test-provider": FakeLLMProvider,
        },
    )

    config = LLMProviderConfig(
        provider="test-provider",
    )

    provider = factory.create(config)

    assert isinstance(
        provider,
        LLMProvider,
    )