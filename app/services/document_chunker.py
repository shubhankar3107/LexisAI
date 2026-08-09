from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    content: str
    chunk_index: int


class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[TextChunk]:
        if not text:
            return []

        chunks: list[TextChunk] = []

        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self._chunk_size

            content = text[start:end].strip()

            if content:
                chunks.append(
                    TextChunk(
                        content=content,
                        chunk_index=chunk_index,
                    )
                )

                chunk_index += 1

            if end >= len(text):
                break

            start = end - self._chunk_overlap

        return chunks
