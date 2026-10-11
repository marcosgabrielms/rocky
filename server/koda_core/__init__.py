"""Contratos puros do núcleo do Koda."""

from .contracts import (
    CoreContextMessage,
    CoreDecision,
    DeviceContext,
    InteractionRequest,
    InteractionResult,
    UsageRecord,
)
from .core import ConversationProcessor, KodaCore
from .conversation_adapter import ConversationCoreAdapter

__all__ = [
    "ConversationProcessor",
    "ConversationCoreAdapter",
    "CoreContextMessage",
    "CoreDecision",
    "DeviceContext",
    "InteractionRequest",
    "InteractionResult",
    "KodaCore",
    "UsageRecord",
]
