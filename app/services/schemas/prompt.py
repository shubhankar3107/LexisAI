from dataclasses import dataclass


@dataclass(frozen=True)
class PromptRequest:
    query: str
    context: str