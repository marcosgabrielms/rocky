"""Fronteira de integração para futuros modelos de linguagem."""

from services.llm.base import ConversationMessage, LLMClient, LLMError, LLMResult, StreamingLLMClient
from services.llm.chatgpt_plan import ChatGPTPlanLLMClient
from services.llm.factory import build_llm_client
from services.llm.fallback import FallbackLLMClient
from services.llm.gemini import GeminiLLMClient
from services.llm.mock import MockLLMClient
from services.llm.openrouter import OpenRouterLLMClient
from services.llm.sentence_buffer import SentenceBuffer

__all__ = [
    "ConversationMessage",
    "ChatGPTPlanLLMClient",
    "FallbackLLMClient",
    "GeminiLLMClient",
    "LLMClient",
    "LLMError",
    "LLMResult",
    "MockLLMClient",
    "OpenRouterLLMClient",
    "SentenceBuffer",
    "StreamingLLMClient",
    "build_llm_client",
]
