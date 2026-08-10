from dataclasses import dataclass, field


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMRequest:
    messages: list[LLMMessage]
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str | None = None
    usage: dict[str, int] = field(default_factory=dict)