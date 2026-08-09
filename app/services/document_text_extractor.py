from __future__ import annotations
from abc import ABC, abstractmethod
from typing import BinaryIO
from dataclasses import dataclass


class DocumentTextExtractor(ABC):

    @abstractmethod
    def extract(
        self,
        file: BinaryIO,
    ) -> list[PageText]:
        """Extract text and page count from a document."""
        raise NotImplementedError

@dataclass(frozen=True)
class PageText:
    page_number: int
    content: str
