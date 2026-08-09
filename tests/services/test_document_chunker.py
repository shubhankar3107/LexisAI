from app.services.document_chunker import DocumentChunker
import pytest


def test_chunk_empty_text_returns_empty_list():
    chunker = DocumentChunker()

    result = chunker.chunk("")

    assert result == []


def test_chunk_text_shorter_than_chunk_size_returns_one_chunk():
    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    result = chunker.chunk("Hello world")

    assert len(result) == 1

    assert result[0].content == "Hello world"
    assert result[0].chunk_index == 0


def test_chunk_text_exactly_chunk_size_returns_one_chunk():
    chunker = DocumentChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    result = chunker.chunk("1234567890")

    assert len(result) == 1
    assert result[0].content == "1234567890"
    assert result[0].chunk_index == 0


def test_chunk_text_creates_multiple_chunks_with_overlap():
    chunker = DocumentChunker(
        chunk_size=10,
        chunk_overlap=2,
    )

    result = chunker.chunk("12345678901234567890")

    assert len(result) == 3

    assert result[0].content == "1234567890"
    assert result[1].content == "9012345678"
    assert result[2].content == "7890"

    assert result[0].chunk_index == 0
    assert result[1].chunk_index == 1
    assert result[2].chunk_index == 2


def test_chunk_indices_are_sequential():
    chunker = DocumentChunker(
        chunk_size=5,
        chunk_overlap=1,
    )

    result = chunker.chunk("123456789012345")

    assert [chunk.chunk_index for chunk in result] == [
        0,
        1,
        2,
        3,
    ]


def test_chunk_strips_leading_and_trailing_whitespace():
    chunker = DocumentChunker(
        chunk_size=100,
        chunk_overlap=10,
    )

    result = chunker.chunk("   Hello world   ")

    assert len(result) == 1
    assert result[0].content == "Hello world"
    assert result[0].chunk_index == 0


import pytest

from app.services.document_chunker import DocumentChunker


def test_chunk_size_must_be_greater_than_zero():
    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than 0",
    ):
        DocumentChunker(chunk_size=0)

    with pytest.raises(
        ValueError,
        match="chunk_size must be greater than 0",
    ):
        DocumentChunker(chunk_size=-1)



def test_chunk_overlap_cannot_be_negative():
    with pytest.raises(
        ValueError,
        match="chunk_overlap cannot be negative",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=-1,
        )


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=100,
        )

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        DocumentChunker(
            chunk_size=100,
            chunk_overlap=101,
        )


def test_chunk_whitespace_only_text_returns_empty_list():
    chunker = DocumentChunker()

    result = chunker.chunk("     \n\t   ")

    assert result == []