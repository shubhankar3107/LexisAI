from io import BytesIO

import pytest

from app.storage.local_file_storage import LocalFileStorage


def test_save_file(tmp_path):
    storage = LocalFileStorage(tmp_path)

    file = BytesIO(b"LexisAI test content")

    storage.save(
        file,
        "documents/test/file.pdf",
    )

    saved_file = tmp_path / "documents/test/file.pdf"

    assert saved_file.exists()
    assert saved_file.read_bytes() == b"LexisAI test content"


def test_save_rejects_path_traversal(tmp_path):
    storage = LocalFileStorage(tmp_path)

    file = BytesIO(b"malicious content")

    with pytest.raises(ValueError):
        storage.save(
            file,
            "../../outside/file.pdf",
        )


def test_delete_file(tmp_path):
    storage = LocalFileStorage(tmp_path)

    file = BytesIO(b"LexisAI test content")

    storage.save(
        file,
        "documents/test/file.pdf",
    )

    storage.delete(
        "documents/test/file.pdf",
    )

    assert not (tmp_path / "documents/test/file.pdf").exists()


def test_delete_rejects_path_traversal(tmp_path):
    storage = LocalFileStorage(tmp_path)

    with pytest.raises(ValueError):
        storage.delete(
            "../../outside/file.pdf",
        )
