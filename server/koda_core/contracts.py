"""Contratos independentes de providers e dispositivos do Koda."""

from __future__ import annotations

from dataclasses import dataclass, field


Metadata = dict[str, object]
Action = dict[str, object]


@dataclass(frozen=True)
class InteractionRequest:
    """Entrada normalizada para uma interação do Koda."""

    user_id: str
    device_id: str
    input_type: str
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class UsageRecord:
    """Métricas de uso devolvidas por um provider, quando disponíveis."""

    provider: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class CoreContextMessage:
    """Mensagem histórica mínima necessária para continuar uma decisão de LLM."""

    role: str
    content: str


@dataclass(frozen=True)
class InteractionResult:
    """Resultado de uma interação, sem acoplamento ao transporte físico."""

    text: str
    interaction_state: str
    actions: list[Action] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    usage: UsageRecord | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class CoreDecision:
    """Decisão pura que separa resposta direta de continuação por LLM."""

    kind: str
    text: str
    device_id: str
    direct_result: InteractionResult | None = None
    context: tuple[CoreContextMessage, ...] = ()
    user_id: str = ""


@dataclass(frozen=True)
class DeviceContext:
    """Contexto genérico de um dispositivo conectado ao Koda."""

    device_id: str
    device_type: str
    capabilities: set[str] = field(default_factory=set)
    metadata: Metadata = field(default_factory=dict)
