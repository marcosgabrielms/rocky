"""Fronteira de integração para futuros modelos de linguagem."""

from services.llm.base import LLMClient, LLMResult
from services.llm.mock import MockLLMClient

__all__ = ["LLMClient", "LLMResult", "MockLLMClient"]
