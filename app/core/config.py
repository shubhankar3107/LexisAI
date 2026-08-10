from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(alias="APP_NAME")
    app_version: str = Field(alias="APP_VERSION")
    debug: bool = Field(alias="DEBUG")

    database_url: str = Field(alias="DATABASE_URL")
    test_database_url: str = Field(alias="TEST_DATABASE_URL")

    upload_directory: Path = Field(alias="UPLOAD_DIRECTORY")


    qdrant_url: str = Field(
        default="http://localhost:6333",
        alias="QDRANT_URL",
    )

    qdrant_api_key: str | None = Field(
        default=None,
        alias="QDRANT_API_KEY",
    )
    qdrant_timeout: float = Field(
        default=10.0,
        alias="QDRANT_TIMEOUT",
    )
    qdrant_collection_prefix: str = Field(
        default="lexisai",
        alias="QDRANT_COLLECTION_PREFIX",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_device: str = Field(
        default="mps",
        alias="EMBEDDING_DEVICE",
    )

    llm_provider: str | None = Field(
        default=None,
        alias="LLM_PROVIDER",
    )

    llm_base_url: str | None = Field(
        default=None,
        alias="LLM_BASE_URL",
    )

    llm_api_key: str | None = Field(
        default=None,
        alias="LLM_API_KEY",
    )

    llm_model: str | None = Field(
        default=None,
        alias="LLM_MODEL",
    )

    llm_timeout: float = Field(
        default=60.0,
        alias="LLM_TIMEOUT",
    )


settings = Settings()