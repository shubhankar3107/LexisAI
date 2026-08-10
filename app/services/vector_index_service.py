from __future__ import annotations

import uuid

from app.db.unit_of_work import UnitOfWork
from app.enums.vector_index_status import VectorIndexStatus
from app.models.vector_index import VectorIndex
from app.repositories.vector_index_repository import VectorIndexRepository
from app.services.embedding_identity_service import EmbeddingIdentityService
from app.services.embedding_registry import EmbeddingRegistry
from app.services.schemas.embedding import EmbeddingIdentitySpec


class IndexNotFoundError(Exception):
    """Raised when a requested vector index does not exist."""


class InvalidIndexTransitionError(Exception):
    """Raised when an invalid vector-index lifecycle transition is requested."""


class VectorIndexService:
    """Application service for vector-index lifecycle management."""

    def __init__(
        self,
        vector_index_repository: VectorIndexRepository,
        embedding_registry: EmbeddingRegistry,
        embedding_identity_service: EmbeddingIdentityService,
        unit_of_work: UnitOfWork,
    ):
        self._vector_index_repository = vector_index_repository
        self._embedding_registry = embedding_registry
        self._embedding_identity_service = embedding_identity_service
        self._unit_of_work = unit_of_work

    def create_index(
        self,
        organization_id: uuid.UUID,
        name: str,
        profile_name: str,
    ) -> VectorIndex:
        """Create a new BUILDING vector index."""

        profile = self._embedding_registry.get(
            profile_name,
        )

        specification = EmbeddingIdentitySpec(
            profile_name=profile.name,
            provider=profile.provider,
            model=profile.model,
            model_version=profile.model_version,
            dimensions=profile.dimensions,
        )

        try:
            embedding_identity = (
                self._embedding_identity_service.resolve(
                    specification,
                )
            )

            vector_index = VectorIndex(
                id=uuid.uuid4(),
                organization_id=organization_id,
                profile_name=profile.name,
                embedding_identity_id=embedding_identity.id,
                name=name,
                status=VectorIndexStatus.BUILDING,
            )

            self._vector_index_repository.add(
                vector_index,
            )

            self._unit_of_work.commit()

            return vector_index

        except Exception:
            self._unit_of_work.rollback()
            raise

    def get_index(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> VectorIndex:
        vector_index = self._vector_index_repository.get_by_id(
            index_id,
            organization_id=organization_id,
        )

        if vector_index is None:
            raise IndexNotFoundError(
                f"Vector index '{index_id}' not found",
            )

        return vector_index

    def require_building_index(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> VectorIndex:
        vector_index = self.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )

        if vector_index.status is not VectorIndexStatus.BUILDING:
            raise InvalidIndexTransitionError(
                "Vector index must be in BUILDING state "
                f"for indexing, but is "
                f"{vector_index.status.value}",
            )

        return vector_index

    def get_active_index(
        self,
        organization_id: uuid.UUID,
        profile_name: str,
    ) -> VectorIndex | None:
        return self._vector_index_repository.get_active(
            organization_id=organization_id,
            profile_name=profile_name,
        )

    def mark_ready(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> VectorIndex:
        vector_index = self.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )

        self._require_transition(
            vector_index,
            VectorIndexStatus.READY,
        )

        try:
            result = self._vector_index_repository.mark_ready(
                vector_index,
            )

            self._unit_of_work.commit()

            return result

        except Exception:
            self._unit_of_work.rollback()
            raise

    def mark_failed(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> VectorIndex:
        vector_index = self.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )

        self._require_transition(
            vector_index,
            VectorIndexStatus.FAILED,
        )

        try:
            result = self._vector_index_repository.mark_failed(
                vector_index,
            )

            self._unit_of_work.commit()

            return result

        except Exception:
            self._unit_of_work.rollback()
            raise

    def activate_index(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> VectorIndex:
        vector_index = self.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )

        self._require_transition(
            vector_index,
            VectorIndexStatus.ACTIVE,
        )

        try:
            result = self._vector_index_repository.activate(
                vector_index,
            )

            self._unit_of_work.commit()

            return result

        except Exception:
            self._unit_of_work.rollback()
            raise

    def deprecate_index(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID,
    ) -> VectorIndex:
        vector_index = self.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )

        self._require_transition(
            vector_index,
            VectorIndexStatus.DEPRECATED,
        )

        try:
            result = self._vector_index_repository.mark_deprecated(
                vector_index,
            )

            self._unit_of_work.commit()

            return result

        except Exception:
            self._unit_of_work.rollback()
            raise

    @staticmethod
    def _require_transition(
        vector_index: VectorIndex,
        target_status: VectorIndexStatus,
    ) -> None:
        allowed_transitions = {
            VectorIndexStatus.BUILDING: {
                VectorIndexStatus.READY,
                VectorIndexStatus.FAILED,
            },
            VectorIndexStatus.READY: {
                VectorIndexStatus.ACTIVE,
            },
            VectorIndexStatus.ACTIVE: {
                VectorIndexStatus.DEPRECATED,
            },
            VectorIndexStatus.DEPRECATED: set(),
            VectorIndexStatus.FAILED: set(),
        }

        if target_status not in allowed_transitions[
            vector_index.status
        ]:
            raise InvalidIndexTransitionError(
                "Invalid vector-index transition: "
                f"{vector_index.status.value} -> "
                f"{target_status.value}",
            )