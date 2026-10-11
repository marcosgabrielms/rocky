"""Testes dos contratos puros de decisão realtime do Koda Core."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import unittest

from koda_core import CoreContextMessage, CoreDecision, InteractionResult, UsageRecord


class CoreDecisionTest(unittest.TestCase):
    def test_direct_decision_preserves_the_complete_direct_result(self) -> None:
        result = InteractionResult(
            text="Agora são 13:04.",
            interaction_state="idle",
            actions=[{"type": "show_text", "line1": "Agora sao"}],
        )

        decision = CoreDecision(
            kind="deterministic",
            text="horas",
            device_id="koda-01",
            direct_result=result,
        )

        self.assertEqual(decision.kind, "deterministic")
        self.assertEqual(decision.text, "horas")
        self.assertEqual(decision.device_id, "koda-01")
        self.assertIs(decision.direct_result, result)
        self.assertEqual(decision.direct_result.actions[0]["type"], "show_text")
        self.assertEqual(decision.context, ())

    def test_llm_decision_preserves_context_without_a_direct_response(self) -> None:
        context = (
            CoreContextMessage(role="user", content="Quem criou o Linux?"),
            CoreContextMessage(role="assistant", content="Linus Torvalds."),
        )

        decision = CoreDecision(
            kind="llm",
            text="E de onde ele é?",
            device_id="koda-01",
            context=context,
        )

        self.assertIsNone(decision.direct_result)
        self.assertEqual(decision.context, context)
        self.assertEqual(decision.context[0].role, "user")

    def test_decision_is_frozen(self) -> None:
        decision = CoreDecision(kind="empty", text="", device_id="koda-01")

        with self.assertRaises(FrozenInstanceError):
            decision.kind = "llm"  # type: ignore[misc]

    def test_direct_result_defaults_remain_independent_between_decisions(self) -> None:
        first = CoreDecision(
            kind="wake",
            text="Koda",
            device_id="koda-01",
            direct_result=InteractionResult(text="Koda", interaction_state="attention"),
        )
        second = CoreDecision(
            kind="wake",
            text="Koda",
            device_id="koda-02",
            direct_result=InteractionResult(text="Koda", interaction_state="attention"),
        )

        first.direct_result.actions.append({"type": "expression", "value": "attention"})

        self.assertEqual(second.direct_result.actions, [])

    def test_decision_supports_optional_provider_model_and_usage_in_direct_result(self) -> None:
        usage = UsageRecord(provider="chatgpt-plan", model="gpt-5.6-terra", total_tokens=21)
        decision = CoreDecision(
            kind="deterministic",
            text="",
            device_id="koda-01",
            direct_result=InteractionResult(
                text="Resposta pronta.",
                interaction_state="idle",
                provider=usage.provider,
                model=usage.model,
                usage=usage,
            ),
        )

        self.assertEqual(decision.direct_result.provider, "chatgpt-plan")
        self.assertEqual(decision.direct_result.model, "gpt-5.6-terra")
        self.assertIs(decision.direct_result.usage, usage)

    def test_decision_supports_basic_dataclass_serialization(self) -> None:
        decision = CoreDecision(
            kind="llm",
            text="Olá.",
            device_id="koda-01",
            context=(CoreContextMessage(role="user", content="Olá."),),
        )

        serialized = asdict(decision)

        self.assertEqual(serialized["kind"], "llm")
        self.assertEqual(serialized["context"], ({"role": "user", "content": "Olá."},))
        self.assertIsNone(serialized["direct_result"])

    def test_contract_source_has_no_runtime_transport_dependencies(self) -> None:
        source = (Path(__file__).parent / "koda_core" / "contracts.py").read_text(encoding="utf-8")

        for forbidden_dependency in ("services.conversation", "fastapi", "esp32", "piper", "audio_queue"):
            self.assertNotIn(forbidden_dependency, source.casefold())


if __name__ == "__main__":
    unittest.main()
