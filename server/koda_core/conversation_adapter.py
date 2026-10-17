"""Ponte isolada entre a conversa legada e os contratos do Koda Core."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .contracts import (
    Action,
    CoreContextMessage,
    CoreDecision,
    InteractionRequest,
    InteractionResult,
    UsageRecord,
)


class ConversationCoreAdapter:
    """Converte decisões legadas sem expor seus objetos aos contratos do Core."""

    def __init__(
        self,
        conversation_manager: object,
        *,
        decision_factory: Callable[..., object] | None = None,
        context_message_factory: Callable[..., object] | None = None,
        kind_factory: Callable[[str], object] | None = None,
    ) -> None:
        if decision_factory is None or context_message_factory is None or kind_factory is None:
            from services.conversation import InteractionDecision, InteractionKind
            from services.llm.base import ConversationMessage

            decision_factory = decision_factory or InteractionDecision
            context_message_factory = context_message_factory or ConversationMessage
            kind_factory = kind_factory or InteractionKind

        self._conversation_manager = conversation_manager
        self._decision_factory = decision_factory
        self._context_message_factory = context_message_factory
        self._kind_factory = kind_factory

    def classify(
        self,
        request: InteractionRequest,
        *,
        received_at: float | None = None,
        interaction_mode: object | None = None,
        low_confidence: bool = False,
    ) -> CoreDecision:
        """Classifica uma vez no legado e devolve somente dados puros do Core."""
        decision = self._conversation_manager.classify(
            request.text,
            request.device_id,
            user_id=request.user_id,
            received_at=received_at,
            interaction_mode=interaction_mode,
            low_confidence=low_confidence,
        )
        return self._to_core_decision(decision, request.user_id)

    def build_direct_result(self, decision: CoreDecision) -> InteractionResult:
        """Constrói uma resposta direta pela API legada sem classificar novamente."""
        if decision.direct_result is None:
            raise ValueError("A decisão não possui resposta direta")
        return self.build_response(decision)

    def build_response(self, decision: CoreDecision) -> InteractionResult:
        """Constrói uma resposta completa sem expor a decisão legada ao runtime."""
        response = self._conversation_manager.build_response_from_decision(
            self.to_conversation_decision(decision)
        )
        return self._to_interaction_result(response)

    def build_stream_response(
        self, decision: CoreDecision, *, on_sentence: Callable[..., None] | None = None
    ) -> object:
        """Continua uma resposta em streaming sem classificar novamente."""
        return self._conversation_manager.build_stream_response_from_decision(
            self.to_conversation_decision(decision), on_sentence=on_sentence,
        )

    def to_conversation_decision(self, decision: CoreDecision) -> object:
        """Reconstrói uma decisão legada equivalente para continuidade existente."""
        direct_response = (
            self._to_direct_response(decision.direct_result)
            if decision.direct_result is not None
            else None
        )
        context = tuple(
            self._context_message_factory(role=message.role, content=message.content)
            for message in decision.context
        )
        return self._decision_factory(
            kind=self._kind_factory(decision.kind),
            text=decision.text,
            device_id=decision.device_id,
            direct_response=direct_response,
            context=context,
            user_id=decision.user_id,
        )

    def _to_core_decision(self, decision: object, user_id: str) -> CoreDecision:
        kind = getattr(decision, "kind").value
        direct_response = getattr(decision, "direct_response")
        context = tuple(
            CoreContextMessage(role=message.role, content=message.content)
            for message in getattr(decision, "context")
        )
        return CoreDecision(
            kind=kind,
            text=getattr(decision, "text"),
            device_id=getattr(decision, "device_id"),
            direct_result=(
                self._to_interaction_result(direct_response)
                if direct_response is not None
                else None
            ),
            context=context,
            user_id=getattr(decision, "user_id", "") or user_id,
        )

    @staticmethod
    def _to_interaction_result(response: Mapping[str, object]) -> InteractionResult:
        raw_actions = response.get("actions", [])
        actions: list[Action] = []
        if isinstance(raw_actions, list):
            actions = [dict(action) for action in raw_actions if isinstance(action, Mapping)]

        text = response.get("text", "")
        interaction_state = response.get("interaction_state", "idle")
        provider = response.get("provider")
        model = response.get("model")
        usage = response.get("usage")
        raw_metadata = response.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        return InteractionResult(
            text=text if isinstance(text, str) else "",
            interaction_state=interaction_state if isinstance(interaction_state, str) else "idle",
            actions=actions,
            provider=provider if isinstance(provider, str) else None,
            model=model if isinstance(model, str) else None,
            usage=usage if isinstance(usage, UsageRecord) else None,
            metadata=metadata,
        )

    @staticmethod
    def _to_direct_response(result: InteractionResult) -> dict[str, object]:
        response: dict[str, object] = {
            "text": result.text,
            "interaction_state": result.interaction_state,
            "actions": [dict(action) for action in result.actions],
        }
        if result.metadata:
            response["metadata"] = dict(result.metadata)
        return response
