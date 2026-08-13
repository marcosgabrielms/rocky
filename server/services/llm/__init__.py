"""Fronteira de integração para futuros modelos de linguagem."""

from services.llm.base import LLMClient, LLMError, LLMResult
from services.llm.mock import MockLLMClient
from services.llm.openrouter import OpenRouterLLMClient

__all__ = ["LLMClient", "LLMError", "LLMResult", "MockLLMClient", "OpenRouterLLMClient"]
