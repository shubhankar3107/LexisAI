import uuid

import pytest

from app.services.context_assembler import (
    ContextAssembler,
    ContextAssembly,
    ContextSource,
)
from app.services.schemas.retrieval import RetrievalResult


def make_result(
    content: str,
    score: float = 0.9,
    page_number: int = 1,
    chunk_index: int = 0,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        page_number=page_number,
        chunk_index=chunk_index,
        content=content,
        score=score,
    )


def test_context_assembler_returns_empty_context_for_empty_results():
    assembler = ContextAssembler()

    result = assembler.assemble([])

    assert isinstance(
        result,
        ContextAssembly,
    )

    assert result.text == ""
    assert result.sources == []


def test_context_assembler_assembles_single_result():
    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    retrieval_result = RetrievalResult(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=3,
        chunk_index=7,
        content="The agreement requires thirty days notice.",
        score=0.923456,
    )

    assembler = ContextAssembler()

    result = assembler.assemble(
        [retrieval_result],
    )

    assert result.text == (
        f"[Document {document_id} | "
        f"Chunk {chunk_id} | "
        f"Page 3 | "
        f"Chunk Index 7 | "
        f"Score 0.923456]\n"
        "The agreement requires thirty days notice."
    )


def test_context_assembler_preserves_source_metadata():
    document_id = uuid.uuid4()
    chunk_id = uuid.uuid4()

    retrieval_result = RetrievalResult(
        document_id=document_id,
        chunk_id=chunk_id,
        page_number=5,
        chunk_index=12,
        content="Payment is due within thirty days.",
        score=0.881234,
    )

    assembler = ContextAssembler()

    result = assembler.assemble(
        [retrieval_result],
    )

    assert len(result.sources) == 1

    source = result.sources[0]

    assert isinstance(
        source,
        ContextSource,
    )

    assert source.document_id == str(
        document_id,
    )

    assert source.chunk_id == str(
        chunk_id,
    )

    assert source.page_number == 5
    assert source.chunk_index == 12
    assert source.score == 0.881234


def test_context_assembler_preserves_result_order():
    first = make_result(
        content="First clause.",
        score=0.95,
        page_number=1,
        chunk_index=0,
    )

    second = make_result(
        content="Second clause.",
        score=0.90,
        page_number=2,
        chunk_index=1,
    )

    third = make_result(
        content="Third clause.",
        score=0.85,
        page_number=3,
        chunk_index=2,
    )

    assembler = ContextAssembler()

    result = assembler.assemble(
        [
            first,
            second,
            third,
        ],
    )

    assert result.sources[0].chunk_id == str(
        first.chunk_id,
    )

    assert result.sources[1].chunk_id == str(
        second.chunk_id,
    )

    assert result.sources[2].chunk_id == str(
        third.chunk_id,
    )

    assert result.text.index(
        "First clause.",
    ) < result.text.index(
        "Second clause.",
    )

    assert result.text.index(
        "Second clause.",
    ) < result.text.index(
        "Third clause.",
    )


def test_context_assembler_separates_multiple_results():
    first = make_result(
        content="First clause.",
    )

    second = make_result(
        content="Second clause.",
    )

    assembler = ContextAssembler()

    result = assembler.assemble(
        [
            first,
            second,
        ],
    )

    assert "\n\n" in result.text

    assert "First clause." in result.text
    assert "Second clause." in result.text


def test_context_assembler_applies_character_limit():
    first = make_result(
        content="A" * 100,
    )

    second = make_result(
        content="B" * 100,
    )

    assembler = ContextAssembler(
        max_characters=300,
    )

    result = assembler.assemble(
        [
            first,
            second,
        ],
    )

    assert len(result.text) <= 300

    assert first.content in result.text
    assert second.content not in result.text

    assert len(result.sources) == 1


def test_context_assembler_does_not_truncate_individual_chunks():
    result = make_result(
        content="A" * 100,
    )

    assembler = ContextAssembler(
        max_characters=50,
    )

    assembly = assembler.assemble(
        [result],
    )

    assert assembly.text == ""

    assert assembly.sources == []


def test_context_assembler_rejects_invalid_character_limit():
    with pytest.raises(
        ValueError,
        match="max_characters must be greater than zero",
    ):
        ContextAssembler(
            max_characters=0,
        )

    with pytest.raises(
        ValueError,
        match="max_characters must be greater than zero",
    ):
        ContextAssembler(
            max_characters=-1,
        )


def test_context_assembler_does_not_modify_results():
    first = make_result(
        content="First clause.",
        score=0.95,
    )

    second = make_result(
        content="Second clause.",
        score=0.85,
    )

    original = [
        first,
        second,
    ]

    assembler = ContextAssembler()

    assembler.assemble(original)

    assert original == [
        first,
        second,
    ]


def test_context_assembler_returns_fresh_source_list():
    result = make_result(
        content="Contract clause.",
    )

    assembler = ContextAssembler()

    first = assembler.assemble(
        [result],
    )

    second = assembler.assemble(
        [result],
    )

    assert first.sources == second.sources
    assert first.sources is not second.sources