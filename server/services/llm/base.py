"""Interface independente de fornecedor para modelos de linguagem."""

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class LLMResult:
    text: str


@runtime_checkable
class LLMClient(Protocol):
    def ask(self, message: str, context: Sequence[str] | None = None) -> LLMResult:
        """Retorna uma resposta textual para uma mensagem do usuário."""
