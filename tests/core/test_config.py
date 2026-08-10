from pathlib import Path

from app.core.config import Settings


def make_settings(**overrides):
    values = {
        "APP_NAME": "LexisAI",
        "APP_VERSION": "1.0.0",
        "DEBUG": "true",
        "DATABASE_URL": "sqlite:///test.db",
        "TEST_DATABASE_URL": "sqlite:///test_test.db",
        "UPLOAD_DIRECTORY": "/tmp/lexisai",
    }

    values.update(overrides)

    return Settings(
        _env_file=None,
        **values,
    )


def test_settings_loads_required_application_configuration():
    settings = make_settings()

    assert settings.app_name == "LexisAI"
    assert settings.app_version == "1.0.0"
    assert settings.debug is True


def test_settings_loads_database_configuration():
    settings = make_settings(
        DATABASE_URL="postgresql://app:secret@localhost/lexisai",
        TEST_DATABASE_URL=(
            "postgresql://app:secret@localhost/lexisai_test"
        ),
    )

    assert settings.database_url == (
        "postgresql://app:secret@localhost/lexisai"
    )

    assert settings.test_database_url == (
        "postgresql://app:secret@localhost/lexisai_test"
    )


def test_settings_loads_upload_directory():
    settings = make_settings(
        UPLOAD_DIRECTORY="/var/lib/lexisai/uploads",
    )

    assert settings.upload_directory == Path(
        "/var/lib/lexisai/uploads",
    )


def test_qdrant_settings_have_safe_defaults():
    settings = make_settings()

    assert settings.qdrant_url == (
        "http://localhost:6333"
    )

    assert settings.qdrant_api_key is None
    assert settings.qdrant_timeout == 10.0
    assert settings.qdrant_collection_prefix == "lexisai"


def test_embedding_device_defaults_to_mps():
    settings = make_settings()

    assert settings.embedding_device == "mps"


def test_qdrant_configuration_can_be_overridden():
    settings = make_settings(
        QDRANT_URL="http://qdrant.internal:6333",
        QDRANT_API_KEY="test-api-key",
        QDRANT_TIMEOUT="20",
        QDRANT_COLLECTION_PREFIX="production-lexisai",
    )

    assert settings.qdrant_url == (
        "http://qdrant.internal:6333"
    )

    assert settings.qdrant_api_key == "test-api-key"
    assert settings.qdrant_timeout == 20.0
    assert settings.qdrant_collection_prefix == (
        "production-lexisai"
    )


def test_embedding_device_can_be_overridden():
    settings = make_settings(
        EMBEDDING_DEVICE="cpu",
    )

    assert settings.embedding_device == "cpu"


def test_llm_settings_are_optional_by_default():
    settings = make_settings()

    assert settings.llm_provider is None
    assert settings.llm_base_url is None
    assert settings.llm_api_key is None
    assert settings.llm_model is None
    assert settings.llm_timeout == 60.0


def test_llm_settings_can_be_configured():
    settings = make_settings(
        LLM_PROVIDER="test-provider",
        LLM_BASE_URL="http://localhost:8000/v1",
        LLM_API_KEY="test-api-key",
        LLM_MODEL="test-model",
        LLM_TIMEOUT="30",
    )

    assert settings.llm_provider == "test-provider"
    assert settings.llm_base_url == (
        "http://localhost:8000/v1"
    )
    assert settings.llm_api_key == "test-api-key"
    assert settings.llm_model == "test-model"
    assert settings.llm_timeout == 30.0