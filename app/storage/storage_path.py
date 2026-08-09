from pathlib import Path
import uuid


class StoragePathBuilder:
    def build(
        self,
        document_id: uuid.UUID,
        original_filename: str,
    ) -> tuple[str, str]:
        extension = Path(original_filename).suffix.lower()

        stored_filename = f"{document_id}{extension}"
        storage_path = f"documents/{document_id}/{stored_filename}"

        return stored_filename, storage_path
