from __future__ import annotations

from app.services.schemas.llm import LLMMessage
from app.services.schemas.prompt import PromptRequest


class PromptBuilder:
    """Build model-agnostic prompts for RAG generation."""

    SYSTEM_PROMPT = """You are a precise legal document assistant.

Answer the user's question using only the provided document context.

Rules:
- Do not invent facts that are not supported by the context.
- If the context does not contain enough information to answer the question, say so clearly.
- Distinguish between information explicitly stated in the documents and reasonable interpretation.
- Be concise and directly answer the user's question.
"""

    def build(
        self,
        request: PromptRequest,
    ) -> list[LLMMessage]:
        return [
            LLMMessage(
                role="system",
                content=self.SYSTEM_PROMPT,
            ),
            LLMMessage(
                role="user",
                content=(
                    "Document context:\n\n"
                    f"{request.context}\n\n"
                    "User question:\n\n"
                    f"{request.query}"
                ),
            ),
        ]