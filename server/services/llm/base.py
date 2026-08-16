"""Interface independente de fornecedor para modelos de linguagem."""

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, runtime_checkable


class LLMError(Exception):
    """Erro controlado ao solicitar ou interpretar uma resposta de LLM."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str | None = None


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        """Retorna uma resposta textual para uma mensagem do usuário."""
