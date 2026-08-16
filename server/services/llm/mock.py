"""Implementação determinística do adapter de LLM para testes locais."""

from typing import Sequence

from services.llm.base import ConversationMessage, LLMResult


class MockLLMClient:
    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        del message, context
        return LLMResult(text="Resposta simulada do Rocky.")
