import uuid

import pytest

from app.services.context_assembler import ContextAssembler
from app.services.llm_provider import LLMProvider
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.reranker import Reranker
from app.services.retriever import Retriever
from app.services.schemas.llm import (
    LLMRequest,
    LLMResponse,
)
from app.services.schemas.rag import RAGRequest
from app.services.schemas.retrieval import (
    RetrievalQuery,
    RetrievalResult,
)


class FakeRetriever(Retriever):

    def __init__(self, results):
        self.results = results
        self.received_query = None

    def retrieve(
        self,
        query: RetrievalQuery,
    ) -> list[RetrievalResult]:
        self.received_query = query

        return self.results


class FakeReranker(Reranker):

    def __init__(
        self,
        results=None,
    ):
        self.results = results
        self.received_query = None
        self.received_results = None
        self.received_top_k = None

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        self.received_query = query
        self.received_results = results
        self.received_top_k = top_k

        if self.results is not None:
            return self.results

        return results[:top_k]


class FakeLLMProvider(LLMProvider):

    def __init__(
        self,
        response=None,
    ):
        self.received_request = None
        self.response = response or LLMResponse(
            content=(
                "The contract requires 30 days notice."
            ),
            model="fake-model",
        )

    def generate(
        self,
        request: LLMRequest,
    ) -> LLMResponse:
        self.received_request = request

        return self.response


def make_result(
    *,
    content: str,
    score: float = 0.9,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=uuid.uuid4(),
        chunk_id=uuid.uuid4(),
        page_number=2,
        chunk_index=3,
        content=content,
        score=score,
    )


def make_orchestrator(
    results,
):
    retriever = FakeRetriever(results)
    reranker = FakeReranker()

    llm_provider = FakeLLMProvider()
    llm_service = LLMService(llm_provider)

    orchestrator = RAGOrchestrator(
        retriever=retriever,
        reranker=reranker,
        context_assembler=ContextAssembler(),
        prompt_builder=PromptBuilder(),
        llm_service=llm_service,
    )

    return (
        orchestrator,
        retriever,
        reranker,
        llm_provider,
    )


def test_rag_orchestrator_generates_answer():
    result = make_result(
        content=(
            "The contract may be terminated with "
            "30 days notice."
        ),
    )

    (
        orchestrator,
        _,
        _,
        _,
    ) = make_orchestrator(
        [result],
    )

    request = RAGRequest(
        query="What is the termination notice?",
        organization_id=uuid.uuid4(),
    )

    response = orchestrator.answer(request)

    assert response.answer == (
        "The contract requires 30 days notice."
    )


def test_rag_orchestrator_passes_retrieval_query():
    result = make_result(
        content="30 days notice.",
    )

    (
        orchestrator,
        retriever,
        _,
        _,
    ) = make_orchestrator(
        [result],
    )

    organization_id = uuid.uuid4()
    document_id = uuid.uuid4()

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=organization_id,
        top_k=5,
        document_id=document_id,
        filters={
            "document_type": "contract",
        },
    )

    orchestrator.answer(request)

    assert retriever.received_query is not None

    assert retriever.received_query.query == (
        "What is the notice period?"
    )

    assert (
        retriever.received_query.organization_id
        == organization_id
    )

    assert retriever.received_query.top_k == 5

    assert (
        retriever.received_query.document_id
        == document_id
    )

    assert retriever.received_query.filters == {
        "document_type": "contract",
    }


def test_rag_orchestrator_copies_retrieval_filters():
    result = make_result(
        content="30 days notice.",
    )

    (
        orchestrator,
        retriever,
        _,
        _,
    ) = make_orchestrator(
        [result],
    )

    filters = {
        "document_type": "contract",
    }

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=uuid.uuid4(),
        filters=filters,
    )

    orchestrator.answer(request)

    assert retriever.received_query.filters == filters
    assert retriever.received_query.filters is not filters


def test_rag_orchestrator_passes_results_to_reranker():
    results = [
        make_result(
            content="First result.",
            score=0.8,
        ),
        make_result(
            content="Second result.",
            score=0.7,
        ),
    ]

    (
        orchestrator,
        _,
        reranker,
        _,
    ) = make_orchestrator(
        results,
    )

    request = RAGRequest(
        query="What does the contract say?",
        organization_id=uuid.uuid4(),
        top_k=1,
    )

    orchestrator.answer(request)

    assert reranker.received_query == (
        "What does the contract say?"
    )

    assert reranker.received_results == results
    assert reranker.received_top_k == 1


def test_rag_orchestrator_passes_context_to_llm():
    result = make_result(
        content=(
            "The contract may be terminated with "
            "30 days notice."
        ),
    )

    (
        orchestrator,
        _,
        _,
        llm_provider,
    ) = make_orchestrator(
        [result],
    )

    request = RAGRequest(
        query="What is the termination notice?",
        organization_id=uuid.uuid4(),
    )

    orchestrator.answer(request)

    llm_request = llm_provider.received_request

    assert llm_request is not None
    assert len(llm_request.messages) == 2

    user_message = llm_request.messages[1]

    assert (
        "The contract may be terminated with "
        "30 days notice."
    ) in user_message.content

    assert (
        "What is the termination notice?"
    ) in user_message.content


def test_rag_orchestrator_preserves_sources():
    result = make_result(
        content="30 days notice.",
        score=0.91,
    )

    (
        orchestrator,
        _,
        _,
        _,
    ) = make_orchestrator(
        [result],
    )

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=uuid.uuid4(),
    )

    response = orchestrator.answer(request)

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.document_id == str(
        result.document_id,
    )

    assert source.chunk_id == str(
        result.chunk_id,
    )

    assert source.page_number == result.page_number
    assert source.chunk_index == result.chunk_index
    assert source.score == result.score


def test_rag_orchestrator_handles_empty_retrieval_without_llm():
    (
        orchestrator,
        _,
        _,
        llm_provider,
    ) = make_orchestrator(
        [],
    )

    request = RAGRequest(
        query="What is the termination notice?",
        organization_id=uuid.uuid4(),
    )

    response = orchestrator.answer(request)

    assert response.answer == (
        RAGOrchestrator.NO_CONTEXT_ANSWER
    )

    assert response.sources == []

    assert llm_provider.received_request is None


def test_rag_orchestrator_handles_empty_reranking_without_llm():
    result = make_result(
        content="30 days notice.",
    )

    retriever = FakeRetriever(
        [result],
    )

    reranker = FakeReranker(
        results=[],
    )

    llm_provider = FakeLLMProvider()
    llm_service = LLMService(
        llm_provider,
    )

    orchestrator = RAGOrchestrator(
        retriever=retriever,
        reranker=reranker,
        context_assembler=ContextAssembler(),
        prompt_builder=PromptBuilder(),
        llm_service=llm_service,
    )

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=uuid.uuid4(),
    )

    response = orchestrator.answer(request)

    assert response.answer == (
        RAGOrchestrator.NO_CONTEXT_ANSWER
    )

    assert response.sources == []

    assert llm_provider.received_request is None


def test_rag_orchestrator_propagates_llm_failure():
    class FailingLLMProvider(LLMProvider):

        def generate(
            self,
            request: LLMRequest,
        ) -> LLMResponse:
            raise RuntimeError(
                "Generation provider failed",
            )

    result = make_result(
        content="30 days notice.",
    )

    retriever = FakeRetriever(
        [result],
    )

    reranker = FakeReranker()

    llm_service = LLMService(
        FailingLLMProvider(),
    )

    orchestrator = RAGOrchestrator(
        retriever=retriever,
        reranker=reranker,
        context_assembler=ContextAssembler(),
        prompt_builder=PromptBuilder(),
        llm_service=llm_service,
    )

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=uuid.uuid4(),
    )

    with pytest.raises(
        RuntimeError,
        match="Generation provider failed",
    ):
        orchestrator.answer(request)


def test_rag_orchestrator_does_not_mutate_request():
    result = make_result(
        content="30 days notice.",
    )

    (
        orchestrator,
        _,
        _,
        _,
    ) = make_orchestrator(
        [result],
    )

    filters = {
        "document_type": "contract",
    }

    request = RAGRequest(
        query="What is the notice period?",
        organization_id=uuid.uuid4(),
        top_k=5,
        filters=filters,
    )

    orchestrator.answer(request)

    assert request.query == (
        "What is the notice period?"
    )

    assert request.top_k == 5
    assert request.filters == filters