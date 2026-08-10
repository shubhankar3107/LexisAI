from app.services.prompt_builder import PromptBuilder
from app.services.schemas.llm import LLMMessage
from app.services.schemas.prompt import PromptRequest


def test_prompt_builder_creates_system_and_user_messages():
    builder = PromptBuilder()

    request = PromptRequest(
        query="What is the termination notice?",
        context=(
            "[Document abc | Page 4]\n"
            "The contract may be terminated with 30 days notice."
        ),
    )

    messages = builder.build(request)

    assert len(messages) == 2

    assert isinstance(messages[0], LLMMessage)
    assert isinstance(messages[1], LLMMessage)

    assert messages[0].role == "system"
    assert messages[1].role == "user"


def test_prompt_builder_includes_context():
    builder = PromptBuilder()

    context = (
        "[Document abc | Page 4]\n"
        "The contract may be terminated with 30 days notice."
    )

    request = PromptRequest(
        query="What is the termination notice?",
        context=context,
    )

    messages = builder.build(request)

    assert context in messages[1].content


def test_prompt_builder_includes_query():
    builder = PromptBuilder()

    query = "What is the termination notice?"

    request = PromptRequest(
        query=query,
        context="The contract may be terminated with 30 days notice.",
    )

    messages = builder.build(request)

    assert query in messages[1].content


def test_prompt_builder_includes_system_instructions():
    builder = PromptBuilder()

    request = PromptRequest(
        query="What is the termination notice?",
        context="30 days notice.",
    )

    messages = builder.build(request)

    system_prompt = messages[0].content

    assert "using only the provided document context" in system_prompt
    assert "Do not invent facts" in system_prompt


def test_prompt_builder_does_not_mutate_request():
    builder = PromptBuilder()

    request = PromptRequest(
        query="What is the termination notice?",
        context="30 days notice.",
    )

    original_query = request.query
    original_context = request.context

    builder.build(request)

    assert request.query == original_query
    assert request.context == original_context