"""Testes isolados do adaptador entre conversa legada e Koda Core."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import unittest

from koda_core import ConversationCoreAdapter, InteractionRequest


class LegacyKind(StrEnum):
    EMPTY = "empty"
    WAKE = "wake"
    DETERMINISTIC = "deterministic"
    LLM = "llm"


@dataclass(frozen=True)
class LegacyMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LegacyDecision:
    kind: LegacyKind
    text: str
    device_id: str
    direct_response: dict[str, object] | None = None
    context: tuple[LegacyMessage, ...] = ()
    user_id: str = ""


class StubConversationManager:
    def __init__(self, decision: LegacyDecision) -> None:
        self.decision = decision
        self.classify_calls: list[tuple[str, str, str | None, float | None, object | None, bool]] = []
        self.direct_calls: list[LegacyDecision] = []
        self.stream_calls: list[LegacyDecision] = []

    def classify(
        self,
        text: str,
        device_id: str,
        *,
        user_id: str | None = None,
        received_at: float | None = None,
        interaction_mode: object | None = None,
        low_confidence: bool = False,
    ) -> LegacyDecision:
        self.classify_calls.append((text, device_id, user_id, received_at, interaction_mode, low_confidence))
        return self.decision

    def build_response_from_decision(self, decision: LegacyDecision) -> dict[str, object]:
        self.direct_calls.append(decision)
        return decision.direct_response or {
            "text": "Resposta do modelo.",
            "interaction_state": "idle",
            "actions": [],
        }

    def build_stream_response_from_decision(
        self, decision: LegacyDecision, *, on_sentence: object | None = None
    ) -> LegacyDecision:
        del on_sentence
        self.stream_calls.append(decision)
        return decision


class FailingConversationManager:
    def classify(self, *args: object, **kwargs: object) -> LegacyDecision:
        raise RuntimeError("classify failure")


def make_adapter(manager: object) -> ConversationCoreAdapter:
    return ConversationCoreAdapter(
        manager,
        decision_factory=LegacyDecision,
        context_message_factory=LegacyMessage,
        kind_factory=LegacyKind,
    )


class ConversationCoreAdapterTest(unittest.TestCase):
    def test_classify_delegates_request_text_and_device_id(self) -> None:
        manager = StubConversationManager(
            LegacyDecision(LegacyKind.EMPTY, text="", device_id="koda-01", direct_response={
                "text": "", "interaction_state": "idle", "actions": [],
            })
        )
        request = InteractionRequest("local-user", "koda-01", "voice_transcript", "Koda")

        decision = make_adapter(manager).classify(
            request,
            received_at=12.5,
            interaction_mode="wake",
            low_confidence=True,
        )

        self.assertEqual(manager.classify_calls, [("Koda", "koda-01", "local-user", 12.5, "wake", True)])
        self.assertEqual(decision.device_id, "koda-01")
        self.assertEqual(decision.user_id, "local-user")

    def test_all_legacy_kinds_convert_to_core_decisions(self) -> None:
        direct_response = {"text": "Koda", "interaction_state": "attention", "actions": []}
        for kind in LegacyKind:
            with self.subTest(kind=kind):
                legacy = LegacyDecision(
                    kind,
                    text="Koda",
                    device_id="koda-01",
                    direct_response=None if kind is LegacyKind.LLM else direct_response,
                    context=(LegacyMessage("user", "Olá"),) if kind is LegacyKind.LLM else (),
                )

                decision = make_adapter(StubConversationManager(legacy)).classify(
                    InteractionRequest("local-user", "koda-01", "text", "Koda")
                )

                self.assertEqual(decision.kind, kind.value)
                self.assertEqual(decision.text, "Koda")
                self.assertEqual(decision.device_id, "koda-01")
                self.assertEqual(decision.direct_result is None, kind is LegacyKind.LLM)

    def test_direct_response_converts_to_interaction_result_with_actions(self) -> None:
        legacy = LegacyDecision(
            LegacyKind.DETERMINISTIC,
            text="horas",
            device_id="koda-01",
            direct_response={
                "text": "horas",
                "interaction_state": "idle",
                "actions": [{"type": "show_text", "line1": "Agora sao", "line2": "13:04"}],
            },
        )

        decision = make_adapter(StubConversationManager(legacy)).classify(
            InteractionRequest("local-user", "koda-01", "text", "horas")
        )

        self.assertEqual(decision.direct_result.actions[0]["type"], "show_text")
        self.assertIsNone(decision.direct_result.provider)
        self.assertIsNone(decision.direct_result.usage)

    def test_context_round_trip_uses_explicit_message_conversion(self) -> None:
        legacy = LegacyDecision(
            LegacyKind.LLM,
            text="E de onde ele é?",
            device_id="koda-01",
            context=(
                LegacyMessage("user", "Quem criou o Linux?"),
                LegacyMessage("assistant", "Linus Torvalds."),
            ),
        )
        adapter = make_adapter(StubConversationManager(legacy))

        core_decision = adapter.classify(InteractionRequest("local-user", "koda-01", "text", legacy.text))
        rebuilt = adapter.to_conversation_decision(core_decision)

        self.assertEqual(rebuilt.kind, LegacyKind.LLM)
        self.assertEqual(rebuilt.text, legacy.text)
        self.assertEqual(rebuilt.device_id, legacy.device_id)
        self.assertEqual(rebuilt.context, legacy.context)
        self.assertIsNot(rebuilt.context[0], legacy.context[0])

    def test_direct_decision_round_trip_uses_existing_direct_response_builder(self) -> None:
        direct_response = {
            "text": "Koda",
            "interaction_state": "attention",
            "actions": [{"type": "expression", "value": "attention"}],
        }
        manager = StubConversationManager(
            LegacyDecision(LegacyKind.WAKE, "Koda", "koda-01", direct_response=direct_response)
        )
        adapter = make_adapter(manager)
        decision = adapter.classify(InteractionRequest("local-user", "koda-01", "text", "Koda"))

        result = adapter.build_direct_result(decision)

        self.assertEqual(len(manager.classify_calls), 1)
        self.assertEqual(len(manager.direct_calls), 1)
        self.assertEqual(manager.direct_calls[0].kind, LegacyKind.WAKE)
        self.assertEqual(result.actions, direct_response["actions"])

    def test_llm_decision_can_continue_without_reclassification(self) -> None:
        manager = StubConversationManager(
            LegacyDecision(
                LegacyKind.LLM,
                "Quanto é dois mais dois?",
                "koda-01",
                context=(LegacyMessage("user", "Olá"),),
            )
        )
        adapter = make_adapter(manager)
        decision = adapter.classify(
            InteractionRequest("local-user", "koda-01", "text", "Quanto é dois mais dois?")
        )

        continued = adapter.build_stream_response(decision)

        self.assertEqual(len(manager.classify_calls), 1)
        self.assertEqual(len(manager.stream_calls), 1)
        self.assertEqual(continued.kind, LegacyKind.LLM)
        self.assertEqual(continued.context[0].content, "Olá")

    def test_build_response_continues_llm_without_reclassification(self) -> None:
        manager = StubConversationManager(
            LegacyDecision(LegacyKind.LLM, "Pergunta", "koda-01")
        )
        adapter = make_adapter(manager)
        decision = adapter.classify(InteractionRequest("local-user", "koda-01", "text", "Pergunta"))

        result = adapter.build_response(decision)

        self.assertEqual(len(manager.classify_calls), 1)
        self.assertEqual(len(manager.direct_calls), 1)
        self.assertEqual(result.text, "Resposta do modelo.")

    def test_llm_decision_cannot_be_consumed_as_a_direct_result(self) -> None:
        manager = StubConversationManager(
            LegacyDecision(LegacyKind.LLM, "Pergunta", "koda-01")
        )
        adapter = make_adapter(manager)
        decision = adapter.classify(InteractionRequest("local-user", "koda-01", "text", "Pergunta"))

        with self.assertRaisesRegex(ValueError, "não possui resposta direta"):
            adapter.build_direct_result(decision)

        self.assertEqual(manager.direct_calls, [])

    def test_classification_exceptions_propagate(self) -> None:
        request = InteractionRequest("local-user", "koda-01", "text", "Olá")

        with self.assertRaisesRegex(RuntimeError, "classify failure"):
            make_adapter(FailingConversationManager()).classify(request)


if __name__ == "__main__":
    unittest.main()
