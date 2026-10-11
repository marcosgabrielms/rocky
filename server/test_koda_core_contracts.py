"""Testes dos contratos puros do núcleo do Koda."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import unittest

from koda_core import DeviceContext, InteractionRequest, InteractionResult, UsageRecord


class KodaCoreContractsTest(unittest.TestCase):
    def test_interaction_request_preserves_identity_and_is_frozen(self) -> None:
        request = InteractionRequest(
            user_id="user-01",
            device_id="koda-01",
            input_type="voice",
            text="Que horas são?",
        )

        self.assertEqual(request.user_id, "user-01")
        self.assertEqual(request.device_id, "koda-01")
        self.assertEqual(request.input_type, "voice")
        self.assertEqual(request.text, "Que horas são?")
        with self.assertRaises(FrozenInstanceError):
            request.text = "Outro texto"  # type: ignore[misc]

    def test_request_metadata_defaults_are_not_shared(self) -> None:
        first = InteractionRequest("user-01", "koda-01", "text", "Olá")
        second = InteractionRequest("user-02", "koda-02", "text", "Oi")

        first.metadata["origin"] = "test"

        self.assertEqual(second.metadata, {})

    def test_deterministic_result_can_omit_provider_usage_and_actions(self) -> None:
        result = InteractionResult(text="Agora são 13:04.", interaction_state="idle")

        self.assertEqual(result.text, "Agora são 13:04.")
        self.assertEqual(result.interaction_state, "idle")
        self.assertEqual(result.actions, [])
        self.assertIsNone(result.provider)
        self.assertIsNone(result.model)
        self.assertIsNone(result.usage)

    def test_result_actions_and_metadata_defaults_are_not_shared(self) -> None:
        first = InteractionResult(text="Pronto.", interaction_state="idle")
        second = InteractionResult(text="Certo.", interaction_state="idle")

        first.actions.append({"type": "display"})
        first.metadata["source"] = "test"

        self.assertEqual(second.actions, [])
        self.assertEqual(second.metadata, {})

    def test_result_can_include_provider_model_and_usage(self) -> None:
        usage = UsageRecord(
            provider="chatgpt-plan",
            model="gpt-5.6-terra",
            input_tokens=12,
            output_tokens=8,
            total_tokens=20,
            cached_tokens=3,
            reasoning_tokens=4,
        )
        result = InteractionResult(
            text="Linus Torvalds criou o Linux.",
            interaction_state="idle",
            provider=usage.provider,
            model=usage.model,
            usage=usage,
        )

        self.assertEqual(result.provider, "chatgpt-plan")
        self.assertEqual(result.model, "gpt-5.6-terra")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.input_tokens, 12)
        self.assertEqual(result.usage.output_tokens, 8)
        self.assertEqual(result.usage.total_tokens, 20)
        self.assertEqual(result.usage.cached_tokens, 3)
        self.assertEqual(result.usage.reasoning_tokens, 4)

    def test_usage_record_exposes_only_usage_metrics(self) -> None:
        usage = UsageRecord()

        self.assertIsNone(usage.provider)
        self.assertIsNone(usage.total_tokens)
        self.assertFalse(hasattr(usage, "quota"))

    def test_device_context_defaults_are_not_shared(self) -> None:
        first = DeviceContext(device_id="koda-01", device_type="robot")
        second = DeviceContext(device_id="koda-02", device_type="mobile")

        first.capabilities.add("audio_output")
        first.metadata["room"] = "lab"

        self.assertEqual(first.device_id, "koda-01")
        self.assertEqual(first.device_type, "robot")
        self.assertEqual(second.capabilities, set())
        self.assertEqual(second.metadata, {})

    def test_contracts_support_basic_dataclass_serialization(self) -> None:
        result = InteractionResult(
            text="Olá.",
            interaction_state="idle",
            actions=[{"type": "speak"}],
            usage=UsageRecord(total_tokens=7),
        )

        serialized = asdict(result)

        self.assertEqual(serialized["text"], "Olá.")
        self.assertEqual(serialized["actions"], [{"type": "speak"}])
        self.assertEqual(serialized["usage"]["total_tokens"], 7)


if __name__ == "__main__":
    unittest.main()
