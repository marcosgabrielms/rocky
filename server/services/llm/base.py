"""Interface independente de fornecedor para modelos de linguagem."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


class LLMError(Exception):
    """Erro controlado ao solicitar ou interpretar uma resposta de LLM."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    def ask(self, message: str, context: Sequence[str] | None = None) -> LLMResult:
        """Retorna uma resposta textual para uma mensagem do usuário."""
