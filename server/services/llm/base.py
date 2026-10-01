"""Interface independente de fornecedor para modelos de linguagem."""

from dataclasses import dataclass
from collections.abc import Iterator
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


@runtime_checkable
class StreamingLLMClient(LLMClient, Protocol):
    def stream_response(
        self, message: str, context: Sequence[ConversationMessage] | None = None
    ) -> Iterator[str]:
        """Produz fragmentos textuais de uma única resposta do modelo."""
