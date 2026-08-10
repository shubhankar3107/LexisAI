from unittest.mock import MagicMock

from app.api.routes.rag import (
    get_rag_orchestrator,
    router as rag_router,
)
from app.services.context_assembler import ContextAssembler
from app.services.llm_service import LLMService
from app.services.prompt_builder import PromptBuilder
from app.services.rag_orchestrator import RAGOrchestrator
from app.services.reranker import Reranker
from app.services.retriever import Retriever


def test_get_rag_orchestrator_builds_complete_graph():
    retriever = MagicMock(
        spec=Retriever,
    )

    reranker = MagicMock(
        spec=Reranker,
    )

    context_assembler = ContextAssembler()

    prompt_builder = PromptBuilder()

    llm_service = MagicMock(
        spec=LLMService,
    )

    orchestrator = get_rag_orchestrator(
        retriever=retriever,
        reranker=reranker,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        llm_service=llm_service,
    )

    assert isinstance(
        orchestrator,
        RAGOrchestrator,
    )

    assert orchestrator._retriever is retriever

    assert orchestrator._reranker is reranker

    assert (
        orchestrator._context_assembler
        is context_assembler
    )

    assert (
        orchestrator._prompt_builder
        is prompt_builder
    )

    assert orchestrator._llm_service is llm_service


