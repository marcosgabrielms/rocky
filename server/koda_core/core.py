"""Orquestrador mínimo e independente de transporte do Koda Core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .contracts import Action, InteractionRequest, InteractionResult, Metadata, UsageRecord


class ConversationProcessor(Protocol):
    """Fronteira mínima compatível com o processamento conversacional atual."""

    def build_response(self, text: str, device_id: str) -> Mapping[str, object]:
        """Processa texto já disponível e devolve uma resposta independente de transporte."""


class KodaCore:
    """Coordena uma dependência conversacional sem implementar suas regras."""

    def __init__(self, processor: ConversationProcessor) -> None:
        self._processor = processor

    def handle(self, request: InteractionRequest) -> InteractionResult:
        """Delega a interação e adapta sua resposta ao contrato do Core."""
        response = self._processor.build_response(request.text, request.device_id)
        return self._build_result(request, response)

    @staticmethod
    def _build_result(
        request: InteractionRequest,
        response: Mapping[str, object],
    ) -> InteractionResult:
        response_metadata = response.get("metadata")
        metadata: Metadata = dict(response_metadata) if isinstance(response_metadata, Mapping) else {}
        metadata.update(request.metadata)
        metadata.update(
            {
                "user_id": request.user_id,
                "device_id": request.device_id,
                "input_type": request.input_type,
            }
        )

        raw_actions = response.get("actions", [])
        actions: list[Action] = []
        if isinstance(raw_actions, list):
            actions = [dict(action) for action in raw_actions if isinstance(action, Mapping)]

        usage = response.get("usage")
        provider = response.get("provider")
        model = response.get("model")
        text = response.get("text", "")
        interaction_state = response.get("interaction_state", "idle")
        return InteractionResult(
            text=text if isinstance(text, str) else "",
            interaction_state=interaction_state if isinstance(interaction_state, str) else "idle",
            actions=actions,
            provider=provider if isinstance(provider, str) else None,
            model=model if isinstance(model, str) else None,
            usage=usage if isinstance(usage, UsageRecord) else None,
            metadata=metadata,
        )
