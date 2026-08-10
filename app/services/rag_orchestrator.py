from __future__ import annotations

from app.services.context_assembler import ContextAssembler
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.services.reranker import Reranker
from app.services.retriever import Retriever
from app.services.schemas.llm import LLMRequest
from app.services.schemas.prompt import PromptRequest
from app.services.schemas.rag import (
    RAGRequest,
    RAGResponse,
    RAGSource,
)
from app.services.schemas.retrieval import RetrievalQuery


class RAGOrchestrator:
    """
    Coordinate retrieval, reranking, context assembly,
    prompting, and generation.

    The orchestrator contains no knowledge of a concrete
    LLM provider or generation model.
    """

    NO_CONTEXT_ANSWER = (
        "I could not find enough relevant information "
        "in the provided documents to answer this question."
    )

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        context_assembler: ContextAssembler,
        prompt_builder: PromptBuilder,
        llm_service: LLMService,
    ):
        self._retriever = retriever
        self._reranker = reranker
        self._context_assembler = context_assembler
        self._prompt_builder = prompt_builder
        self._llm_service = llm_service

    def answer(
        self,
        request: RAGRequest,
    ) -> RAGResponse:
        retrieval_query = RetrievalQuery(
            query=request.query,
            organization_id=request.organization_id,
            top_k=request.top_k,
            document_id=request.document_id,
            filters=request.filters.copy(),
        )

        retrieved_results = self._retriever.retrieve(
            retrieval_query,
        )

        if not retrieved_results:
            return RAGResponse(
                answer=self.NO_CONTEXT_ANSWER,
                sources=[],
            )

        reranked_results = self._reranker.rerank(
            query=request.query,
            results=retrieved_results,
            top_k=request.top_k,
        )

        if not reranked_results:
            return RAGResponse(
                answer=self.NO_CONTEXT_ANSWER,
                sources=[],
            )

        context = self._context_assembler.assemble(
            reranked_results,
        )

        if not context.text:
            return RAGResponse(
                answer=self.NO_CONTEXT_ANSWER,
                sources=[],
            )

        prompt_request = PromptRequest(
            query=request.query,
            context=context.text,
        )

        messages = self._prompt_builder.build(
            prompt_request,
        )

        llm_request = LLMRequest(
            messages=messages,
        )

        llm_response = self._llm_service.generate(
            llm_request,
        )

        sources = [
            RAGSource(
                document_id=source.document_id,
                chunk_id=source.chunk_id,
                page_number=source.page_number,
                chunk_index=source.chunk_index,
                score=source.score,
            )
            for source in context.sources
        ]

        return RAGResponse(
            answer=llm_response.content,
            sources=sources,
        )