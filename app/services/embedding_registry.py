from app.services.schemas.embedding import EmbeddingProfile


class EmbeddingProfileNotFoundError(Exception):
    """Raised when an embedding profile does not exist."""


class EmbeddingRegistry:

    def __init__(
        self,
        profiles: dict[str, EmbeddingProfile],
    ):
        self._profiles = profiles.copy()

    def get(
        self,
        name: str,
    ) -> EmbeddingProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise EmbeddingProfileNotFoundError(
                f"Embedding profile '{name}' not found",
            ) from exc