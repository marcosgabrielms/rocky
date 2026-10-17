"""Camada de aplicação para interações textuais independentes de dispositivo."""

from collections.abc import Callable, Mapping

from koda_core import CoreDecision, InteractionRequest, InteractionResult


class KodaInteractionService:
    """Executa interações já transcritas usando a fronteira do Koda Core."""

    def __init__(self, conversation_core_adapter: object) -> None:
        self._conversation_core_adapter = conversation_core_adapter

    def classify(
        self,
        *,
        user_id: str,
        device_id: str,
        input_type: str,
        text: str,
        received_at: float | None = None,
        interaction_mode: object | None = None,
        low_confidence: bool = False,
        metadata: Mapping[str, object] | None = None,
    ) -> CoreDecision:
        """Cria o contrato de entrada e classifica uma única vez."""
        request = InteractionRequest(
            user_id=user_id,
            device_id=device_id,
            input_type=input_type,
            text=text,
            metadata=dict(metadata or {}),
        )
        return self._conversation_core_adapter.classify(
            request,
            received_at=received_at,
            interaction_mode=interaction_mode,
            low_confidence=low_confidence,
        )

    def build_response(self, decision: CoreDecision) -> InteractionResult:
        return self._conversation_core_adapter.build_response(decision)

    def build_direct_response(self, decision: CoreDecision) -> InteractionResult:
        return self._conversation_core_adapter.build_direct_result(decision)

    def build_stream_response(
        self, decision: CoreDecision, *, on_sentence: Callable[..., None] | None = None
    ) -> object:
        return self._conversation_core_adapter.build_stream_response(decision, on_sentence=on_sentence)
