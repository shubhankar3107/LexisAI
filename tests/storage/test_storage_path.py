import uuid

from app.storage.storage_path import StoragePathBuilder


def test_build_storage_path():
    document_id = uuid.uuid4()

    stored_filename, storage_path = StoragePathBuilder().build(
        document_id,
        "Employment Agreement.PDF",
    )

    assert stored_filename == f"{document_id}.pdf"
    assert storage_path == (f"documents/{document_id}/{document_id}.pdf")
