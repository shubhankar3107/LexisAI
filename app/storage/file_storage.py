from abc import ABC, abstractmethod
from typing import BinaryIO


class FileStorage(ABC):

    @abstractmethod
    def save(
        self,
        file: BinaryIO,
        path: str,
    ) -> None:
        """Persist a binary file at the specified path."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, path: str) -> None:
        """Delete a file from the storage."""
        raise NotImplementedError

    @abstractmethod
    def read(self, path: str) -> BinaryIO:
        """Read a binary file from the specified path."""
        raise NotImplementedError
