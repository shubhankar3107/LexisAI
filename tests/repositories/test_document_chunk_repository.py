import uuid

from app.models.document_chunk import DocumentChunk
from app.repositories.document_chunk_repository import DocumentChunkRepository


class FakeSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)


def test_add_adds_chunk_to_session():
    session = FakeSession()
    repository = DocumentChunkRepository(session)

    document_id = uuid.uuid4()

    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=0,
        page_number=1,
        content="Contract text",
    )

    result = repository.add(chunk)

    assert result is chunk
    assert session.added == [chunk]


def test_delete_by_document_id_executes_delete_statement():
    session = FakeSession()
    session.executed = []

    def execute(statement):
        session.executed.append(statement)

    session.execute = execute

    repository = DocumentChunkRepository(session)

    document_id = uuid.uuid4()

    repository.delete_by_document_id(document_id)

    assert len(session.executed) == 1

    statement = session.executed[0]

    compiled = str(statement)

    assert "DELETE" in compiled.upper()
    assert "document_chunks" in compiled


def test_delete_by_document_id_returns_none():
    session = FakeSession()
    session.executed = []

    def execute(statement):
        session.executed.append(statement)

    session.execute = execute

    repository = DocumentChunkRepository(session)

    result = repository.delete_by_document_id(uuid.uuid4())

    assert result is None


def test_delete_by_document_id_targets_correct_document():
    session = FakeSession()
    session.executed = []

    def execute(statement):
        session.executed.append(statement)

    session.execute = execute

    repository = DocumentChunkRepository(session)

    document_id = uuid.uuid4()

    repository.delete_by_document_id(document_id)

    statement = session.executed[0]

    compiled = statement.compile(
        compile_kwargs={"literal_binds": True},
    )

    sql = str(compiled)

    assert document_id.hex in sql


def test_delete_by_document_id_filters_by_document_id():
    session = FakeSession()
    session.executed = []

    def execute(statement):
        session.executed.append(statement)

    session.execute = execute

    repository = DocumentChunkRepository(session)

    document_id = uuid.uuid4()

    repository.delete_by_document_id(document_id)

    statement = session.executed[0]

    compiled = statement.compile(
        compile_kwargs={"literal_binds": True},
    )

    sql = str(compiled).lower()

    assert "document_chunks.document_id" in sql
    assert "where" in sql
    assert document_id.hex in sql


def test_list_by_document_id_returns_chunks_in_chunk_order():
    document_id = uuid.uuid4()

    chunk_1 = DocumentChunk(
        document_id=document_id,
        chunk_index=1,
        page_number=1,
        content="Second chunk",
    )

    chunk_0 = DocumentChunk(
        document_id=document_id,
        chunk_index=0,
        page_number=1,
        content="First chunk",
    )

    class FakeResult:
        def scalars(self):
            return self

        def all(self):
            return [chunk_0, chunk_1]

    class QuerySession:
        def __init__(self):
            self.statement = None

        def execute(self, statement):
            self.statement = statement
            return FakeResult()

    session = QuerySession()

    repository = DocumentChunkRepository(session)

    result = repository.list_by_document_id(document_id)

    assert result == [chunk_0, chunk_1]

    sql = str(
        session.statement.compile(
            compile_kwargs={"literal_binds": True},
        )
    ).lower()

    assert "document_chunks.document_id" in sql
    assert "order by document_chunks.chunk_index asc" in sql
    assert document_id.hex in sql