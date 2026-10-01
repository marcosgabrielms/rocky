"""Fronteira de integração para futuros modelos de linguagem."""

from services.llm.base import ConversationMessage, LLMClient, LLMError, LLMResult, StreamingLLMClient
from services.llm.mock import MockLLMClient
from services.llm.openrouter import OpenRouterLLMClient
from services.llm.sentence_buffer import SentenceBuffer

__all__ = [
    "ConversationMessage",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "MockLLMClient",
    "OpenRouterLLMClient",
    "SentenceBuffer",
    "StreamingLLMClient",
]
