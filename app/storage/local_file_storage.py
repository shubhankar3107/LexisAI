from pathlib import Path
from typing import BinaryIO

from app.storage.file_storage import FileStorage


class LocalFileStorage(FileStorage):
    def __init__(self, base_path: Path):
        self._base_path = base_path

    def _resolve_path(self, relative_path: str | Path) -> Path:
        base_path = self._base_path.resolve()
        destination = (base_path / relative_path).resolve()

        try:
            destination.relative_to(base_path)
        except ValueError as exc:
            raise ValueError(
                "Storage path escapes the configured base directory"
            ) from exc

        return destination

    def save(
        self,
        file: BinaryIO,
        relative_path: str,
    ) -> Path:
        destination = self._resolve_path(relative_path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open("wb") as output:
            while chunk := file.read(1024 * 1024):
                output.write(chunk)

        return destination

    def delete(
        self,
        path: str,
    ) -> None:
        destination = self._resolve_path(path)

        if destination.exists():
            destination.unlink()

    def read(
        self,
        path: str,
    ) -> BinaryIO:
        destination = self._resolve_path(path)

        if not destination.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not destination.is_file():
            raise ValueError(f"Storage path is not a file: {path}")

        return destination.open("rb")
