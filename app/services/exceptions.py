class DocumentNotFoundError(Exception):
    pass


class DocumentFileNotFoundError(Exception):
    pass


class DocumentProcessingStateError(Exception):
    pass


class LLMProviderNotFoundError(Exception):
    """Raised when a requested LLM provider is not registered."""
    pass