from __future__ import annotations

import uuid

import pytest

from app.enums.vector_index_status import VectorIndexStatus
from app.models.embedding_identity import EmbeddingIdentity
from app.models.vector_index import VectorIndex
from app.services.embedding_registry import (
    EmbeddingProfileNotFoundError,
)
from app.services.schemas.embedding import (
    EmbeddingIdentitySpec,
    EmbeddingProfile,
)
from app.services.vector_index_service import (
    IndexNotFoundError,
    InvalidIndexTransitionError,
    VectorIndexService,
)


class FakeEmbeddingRegistry:
    def __init__(self):
        self.profile = EmbeddingProfile(
            name="legal-general",
            provider="huggingface",
            model="Qwen3-Embedding-8B",
            model_version="v1",
            dimensions=4096,
        )

    def get(
        self,
        name: str,
    ) -> EmbeddingProfile:
        if name != self.profile.name:
            raise EmbeddingProfileNotFoundError(
                f"Embedding profile '{name}' not found",
            )

        return self.profile


class FakeEmbeddingIdentityService:
    def __init__(self):
        self.identity: EmbeddingIdentity | None = None

    def resolve(
        self,
        specification: EmbeddingIdentitySpec,
    ) -> EmbeddingIdentity:
        self.identity = EmbeddingIdentity(
            id=uuid.uuid4(),
            profile_name=specification.profile_name,
            provider=specification.provider,
            model=specification.model,
            model_version=specification.model_version,
            dimensions=specification.dimensions,
            fingerprint=specification.fingerprint,
        )

        return self.identity


class FakeVectorIndexRepository:
    def __init__(
        self,
        vector_index: VectorIndex | None = None,
    ):
        self.vector_index = vector_index

        self.add_called = False
        self.added_index: VectorIndex | None = None

        self.mark_ready_called = False
        self.mark_failed_called = False
        self.activate_called = False
        self.mark_deprecated_called = False

    def add(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        self.add_called = True
        self.added_index = vector_index
        self.vector_index = vector_index

        return vector_index

    def get_by_id(
        self,
        index_id: uuid.UUID,
        organization_id: uuid.UUID | None = None,
    ) -> VectorIndex | None:
        if self.vector_index is None:
            return None

        if self.vector_index.id != index_id:
            return None

        if (
            organization_id is not None
            and self.vector_index.organization_id != organization_id
        ):
            return None

        return self.vector_index

    def get_active(
        self,
        organization_id: uuid.UUID,
        profile_name: str,
    ) -> VectorIndex | None:
        if self.vector_index is None:
            return None

        if self.vector_index.organization_id != organization_id:
            return None

        if self.vector_index.profile_name != profile_name:
            return None

        if self.vector_index.status != VectorIndexStatus.ACTIVE:
            return None

        return self.vector_index

    def mark_ready(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        self.mark_ready_called = True
        vector_index.status = VectorIndexStatus.READY

        return vector_index

    def mark_failed(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        self.mark_failed_called = True
        vector_index.status = VectorIndexStatus.FAILED

        return vector_index

    def activate(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        self.activate_called = True
        vector_index.status = VectorIndexStatus.ACTIVE

        return vector_index

    def mark_deprecated(
        self,
        vector_index: VectorIndex,
    ) -> VectorIndex:
        self.mark_deprecated_called = True
        vector_index.status = VectorIndexStatus.DEPRECATED

        return vector_index


class FakeUnitOfWork:
    def __init__(self):
        self.commit_called = False
        self.commit_count = 0
        self.rollback_called = False

    def commit(self) -> None:
        self.commit_called = True
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_called = True


def make_vector_index(
    status: VectorIndexStatus,
) -> VectorIndex:
    organization_id = uuid.uuid4()
    embedding_identity_id = uuid.uuid4()

    return VectorIndex(
        id=uuid.uuid4(),
        organization_id=organization_id,
        profile_name="default",
        embedding_identity_id=embedding_identity_id,
        name="test-index",
        status=status,
    )


def make_service(
    vector_index: VectorIndex | None = None,
) -> tuple[
    VectorIndexService,
    FakeVectorIndexRepository,
    FakeUnitOfWork,
]:
    repository = FakeVectorIndexRepository(
        vector_index=vector_index,
    )

    embedding_registry = FakeEmbeddingRegistry()
    embedding_identity_service = FakeEmbeddingIdentityService()

    unit_of_work = FakeUnitOfWork()

    service = VectorIndexService(
        vector_index_repository=repository,
        embedding_registry=embedding_registry,
        embedding_identity_service=embedding_identity_service,
        unit_of_work=unit_of_work,
    )

    return service, repository, unit_of_work


def test_get_index_returns_index_for_correct_organization():
    vector_index = make_vector_index(
        VectorIndexStatus.BUILDING,
    )

    service, _, _ = make_service(vector_index)

    result = service.get_index(
        index_id=vector_index.id,
        organization_id=vector_index.organization_id,
    )

    assert result is vector_index


def test_get_index_raises_when_index_does_not_exist():
    index_id = uuid.uuid4()
    organization_id = uuid.uuid4()

    service, _, _ = make_service()

    with pytest.raises(IndexNotFoundError):
        service.get_index(
            index_id=index_id,
            organization_id=organization_id,
        )


def test_get_index_raises_when_organization_does_not_match():
    vector_index = make_vector_index(
        VectorIndexStatus.BUILDING,
    )

    service, _, _ = make_service(vector_index)

    with pytest.raises(IndexNotFoundError):
        service.get_index(
            index_id=vector_index.id,
            organization_id=uuid.uuid4(),
        )


def test_get_active_index_returns_active_index():
    vector_index = make_vector_index(
        VectorIndexStatus.ACTIVE,
    )

    service, _, _ = make_service(vector_index)

    result = service.get_active_index(
        organization_id=vector_index.organization_id,
        profile_name=vector_index.profile_name,
    )

    assert result is vector_index


def test_get_active_index_returns_none_when_no_active_index_exists():
    vector_index = make_vector_index(
        VectorIndexStatus.READY,
    )

    service, _, _ = make_service(vector_index)

    result = service.get_active_index(
        organization_id=vector_index.organization_id,
        profile_name=vector_index.profile_name,
    )

    assert result is None


def test_mark_ready_transitions_building_index():
    vector_index = make_vector_index(
        VectorIndexStatus.BUILDING,
    )

    service, repository, unit_of_work = make_service(
        vector_index,
    )

    result = service.mark_ready(
        index_id=vector_index.id,
        organization_id=vector_index.organization_id,
    )

    assert result.status == VectorIndexStatus.READY
    assert repository.mark_ready_called is True
    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False


def test_mark_failed_transitions_building_index():
    vector_index = make_vector_index(
        VectorIndexStatus.BUILDING,
    )

    service, repository, unit_of_work = make_service(
        vector_index,
    )

    result = service.mark_failed(
        index_id=vector_index.id,
        organization_id=vector_index.organization_id,
    )

    assert result.status == VectorIndexStatus.FAILED
    assert repository.mark_failed_called is True
    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False


def test_activate_index_transitions_ready_index():
    vector_index = make_vector_index(
        VectorIndexStatus.READY,
    )

    service, repository, unit_of_work = make_service(
        vector_index,
    )

    result = service.activate_index(
        index_id=vector_index.id,
        organization_id=vector_index.organization_id,
    )

    assert result.status == VectorIndexStatus.ACTIVE
    assert repository.activate_called is True
    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False


def test_deprecate_index_transitions_active_index():
    vector_index = make_vector_index(
        VectorIndexStatus.ACTIVE,
    )

    service, repository, unit_of_work = make_service(
        vector_index,
    )

    result = service.deprecate_index(
        index_id=vector_index.id,
        organization_id=vector_index.organization_id,
    )

    assert result.status == VectorIndexStatus.DEPRECATED
    assert repository.mark_deprecated_called is True
    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False


@pytest.mark.parametrize(
    ("current_status", "target_operation"),
    [
        (
            VectorIndexStatus.BUILDING,
            "activate_index",
        ),
        (
            VectorIndexStatus.READY,
            "mark_ready",
        ),
        (
            VectorIndexStatus.READY,
            "mark_failed",
        ),
        (
            VectorIndexStatus.ACTIVE,
            "mark_ready",
        ),
        (
            VectorIndexStatus.ACTIVE,
            "mark_failed",
        ),
        (
            VectorIndexStatus.DEPRECATED,
            "activate_index",
        ),
        (
            VectorIndexStatus.FAILED,
            "activate_index",
        ),
    ],
)
def test_invalid_transitions_are_rejected(
    current_status: VectorIndexStatus,
    target_operation: str,
):
    vector_index = make_vector_index(
        current_status,
    )

    service, repository, unit_of_work = make_service(
        vector_index,
    )

    operation = getattr(
        service,
        target_operation,
    )

    with pytest.raises(InvalidIndexTransitionError):
        operation(
            index_id=vector_index.id,
            organization_id=vector_index.organization_id,
        )

    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False

    assert repository.mark_ready_called is False
    assert repository.mark_failed_called is False
    assert repository.activate_called is False
    assert repository.mark_deprecated_called is False


def test_repository_failure_rolls_back_transaction():
    vector_index = make_vector_index(
        VectorIndexStatus.READY,
    )

    class FailingRepository(FakeVectorIndexRepository):
        def activate(
            self,
            vector_index: VectorIndex,
        ) -> VectorIndex:
            self.activate_called = True

            raise RuntimeError(
                "database failure",
            )

    repository = FailingRepository(
        vector_index=vector_index,
    )

    unit_of_work = FakeUnitOfWork()

    service = VectorIndexService(
        vector_index_repository=repository,
        embedding_registry=FakeEmbeddingRegistry(),
        embedding_identity_service=FakeEmbeddingIdentityService(),
        unit_of_work=unit_of_work,
    )

    with pytest.raises(
        RuntimeError,
        match="database failure",
    ):
        service.activate_index(
            index_id=vector_index.id,
            organization_id=vector_index.organization_id,
        )

    assert repository.activate_called is True
    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is True


def test_create_index_creates_building_index():
    organization_id = uuid.uuid4()

    service, repository, unit_of_work = make_service()

    result = service.create_index(
        organization_id=organization_id,
        name="legal-general-v1",
        profile_name="legal-general",
    )

    assert isinstance(result, VectorIndex)

    assert result.organization_id == organization_id
    assert result.name == "legal-general-v1"
    assert result.profile_name == "legal-general"
    assert result.status == VectorIndexStatus.BUILDING

    assert result.embedding_identity_id is not None

    assert repository.add_called is True
    assert repository.added_index is result

    assert unit_of_work.commit_called is True
    assert unit_of_work.rollback_called is False


def test_create_index_uses_configured_embedding_profile():
    organization_id = uuid.uuid4()

    service, _, _ = make_service()

    result = service.create_index(
        organization_id=organization_id,
        name="legal-index",
        profile_name="legal-general",
    )

    identity_service = service._embedding_identity_service
    identity = identity_service.identity

    assert identity is not None

    assert identity.profile_name == "legal-general"
    assert identity.provider == "huggingface"
    assert identity.model == "Qwen3-Embedding-8B"
    assert identity.model_version == "v1"
    assert identity.dimensions == 4096

    assert result.embedding_identity_id == identity.id


def test_create_index_commits_once():
    organization_id = uuid.uuid4()

    service, _, unit_of_work = make_service()

    service.create_index(
        organization_id=organization_id,
        name="legal-index",
        profile_name="legal-general",
    )

    assert unit_of_work.commit_count == 1
    assert unit_of_work.rollback_called is False


def test_create_index_rolls_back_when_repository_fails():
    class FailingRepository(FakeVectorIndexRepository):
        def add(
            self,
            vector_index: VectorIndex,
        ) -> VectorIndex:
            self.add_called = True
            self.added_index = vector_index

            raise RuntimeError(
                "database failure",
            )

    repository = FailingRepository()

    unit_of_work = FakeUnitOfWork()

    service = VectorIndexService(
        vector_index_repository=repository,
        embedding_registry=FakeEmbeddingRegistry(),
        embedding_identity_service=FakeEmbeddingIdentityService(),
        unit_of_work=unit_of_work,
    )

    organization_id = uuid.uuid4()

    with pytest.raises(
        RuntimeError,
        match="database failure",
    ):
        service.create_index(
            organization_id=organization_id,
            name="legal-index",
            profile_name="legal-general",
        )

    assert repository.add_called is True
    assert repository.added_index is not None
    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is True


def test_create_index_raises_when_profile_does_not_exist():
    class MissingProfileRegistry:
        def get(
            self,
            name: str,
        ):
            raise EmbeddingProfileNotFoundError(
                f"Embedding profile '{name}' not found",
            )

    repository = FakeVectorIndexRepository()
    unit_of_work = FakeUnitOfWork()

    service = VectorIndexService(
        vector_index_repository=repository,
        embedding_registry=MissingProfileRegistry(),
        embedding_identity_service=FakeEmbeddingIdentityService(),
        unit_of_work=unit_of_work,
    )

    organization_id = uuid.uuid4()

    with pytest.raises(
        EmbeddingProfileNotFoundError,
        match="Embedding profile 'missing' not found",
    ):
        service.create_index(
            organization_id=organization_id,
            name="legal-index",
            profile_name="missing",
        )

    assert repository.add_called is False
    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False


def test_require_building_index_returns_building_index():
    vector_index = make_vector_index(
        VectorIndexStatus.BUILDING,
    )

    service, _, unit_of_work = make_service(
        vector_index,
    )

    result = service.require_building_index(
        index_id=vector_index.id,
        organization_id=vector_index.organization_id,
    )

    assert result is vector_index
    assert result.status == VectorIndexStatus.BUILDING

    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False


@pytest.mark.parametrize(
    "status",
    [
        VectorIndexStatus.READY,
        VectorIndexStatus.ACTIVE,
        VectorIndexStatus.FAILED,
        VectorIndexStatus.DEPRECATED,
    ],
)
def test_require_building_index_rejects_non_building_index(
    status: VectorIndexStatus,
):
    vector_index = make_vector_index(
        status,
    )

    service, _, unit_of_work = make_service(
        vector_index,
    )

    with pytest.raises(
        InvalidIndexTransitionError,
        match=(
            "Vector index must be in BUILDING state "
            f"for indexing, but is {status.value}"
        ),
    ):
        service.require_building_index(
            index_id=vector_index.id,
            organization_id=vector_index.organization_id,
        )

    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False


def test_require_building_index_rejects_wrong_organization():
    vector_index = make_vector_index(
        VectorIndexStatus.BUILDING,
    )

    service, _, unit_of_work = make_service(
        vector_index,
    )

    with pytest.raises(IndexNotFoundError):
        service.require_building_index(
            index_id=vector_index.id,
            organization_id=uuid.uuid4(),
        )

    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False



def test_require_building_index_rejects_missing_index():
    service, _, unit_of_work = make_service()

    with pytest.raises(IndexNotFoundError):
        service.require_building_index(
            index_id=uuid.uuid4(),
            organization_id=uuid.uuid4(),
        )

    assert unit_of_work.commit_called is False
    assert unit_of_work.rollback_called is False