from __future__ import annotations

from dataclasses import dataclass

from app.services.schemas.retrieval import RetrievalResult


@dataclass(frozen=True)
class ContextSource:
    """Source metadata associated with an assembled context block."""

    document_id: str
    chunk_id: str
    page_number: int
    chunk_index: int
    score: float


@dataclass(frozen=True)
class ContextAssembly:
    """Structured context prepared for downstream LLM generation."""

    text: str
    sources: list[ContextSource]


class ContextAssembler:
    """
    Assemble retrieval results into deterministic LLM context.

    The assembler does not generate, summarize, rerank, or otherwise
    modify the semantic content of retrieved chunks.
    """

    def __init__(
        self,
        max_characters: int = 12000,
    ):
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero",
            )

        self._max_characters = max_characters

    def assemble(
        self,
        results: list[RetrievalResult],
    ) -> ContextAssembly:
        if not results:
            return ContextAssembly(
                text="",
                sources=[],
            )

        blocks: list[str] = []
        sources: list[ContextSource] = []

        current_length = 0

        for result in results:
            block = self._format_result(
                result,
            )

            separator_length = (
                2
                if blocks
                else 0
            )

            required_length = (
                current_length
                + separator_length
                + len(block)
            )

            if required_length > self._max_characters:
                break

            blocks.append(block)

            sources.append(
                ContextSource(
                    document_id=str(
                        result.document_id,
                    ),
                    chunk_id=str(
                        result.chunk_id,
                    ),
                    page_number=result.page_number,
                    chunk_index=result.chunk_index,
                    score=result.score,
                )
            )

            current_length = required_length

        return ContextAssembly(
            text="\n\n".join(blocks),
            sources=sources,
        )

    @staticmethod
    def _format_result(
        result: RetrievalResult,
    ) -> str:
        return (
            f"[Document {result.document_id} | "
            f"Chunk {result.chunk_id} | "
            f"Page {result.page_number} | "
            f"Chunk Index {result.chunk_index} | "
            f"Score {result.score:.6f}]\n"
            f"{result.content}"
        )