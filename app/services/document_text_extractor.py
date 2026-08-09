from abc import ABC, abstractmethod
from typing import BinaryIO


class DocumentTextExtractor(ABC):

    @abstractmethod
    def extract(
        self,
        file: BinaryIO,
    ) -> tuple[str, int]:
        """Extract text and page count from a document."""
        raise NotImplementedError
