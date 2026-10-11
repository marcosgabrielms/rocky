"""Testes do orquestrador mínimo do Koda Core."""

from __future__ import annotations

import unittest

from koda_core import InteractionRequest, KodaCore, UsageRecord


class StubProcessor:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def build_response(self, text: str, device_id: str) -> dict[str, object]:
        self.calls.append((text, device_id))
        return self.response


class FailingProcessor:
    def build_response(self, text: str, device_id: str) -> dict[str, object]:
        raise RuntimeError("processor failure")


class KodaCoreTest(unittest.TestCase):
    def test_handle_delegates_text_and_device_id(self) -> None:
        processor = StubProcessor({"text": "Olá.", "interaction_state": "idle", "actions": []})
        core = KodaCore(processor)
        request = InteractionRequest("user-01", "koda-01", "voice_transcript", "Olá, Koda")

        result = core.handle(request)

        self.assertEqual(processor.calls, [("Olá, Koda", "koda-01")])
        self.assertEqual(result.text, "Olá.")
        self.assertEqual(result.metadata["user_id"], "user-01")
        self.assertEqual(result.metadata["device_id"], "koda-01")
        self.assertEqual(result.metadata["input_type"], "voice_transcript")

    def test_deterministic_response_does_not_require_llm_metadata(self) -> None:
        processor = StubProcessor(
            {
                "text": "Agora são 13:04.",
                "interaction_state": "idle",
                "actions": [{"type": "show_text"}],
            }
        )

        result = KodaCore(processor).handle(InteractionRequest("user-01", "koda-01", "text", "horas"))

        self.assertEqual(result.actions, [{"type": "show_text"}])
        self.assertIsNone(result.provider)
        self.assertIsNone(result.model)
        self.assertIsNone(result.usage)

    def test_result_preserves_provider_model_and_usage_when_available(self) -> None:
        usage = UsageRecord(provider="chatgpt-plan", model="gpt-5.6-terra", total_tokens=21)
        processor = StubProcessor(
            {
                "text": "Linus Torvalds criou o Linux.",
                "interaction_state": "idle",
                "actions": [],
                "provider": "chatgpt-plan",
                "model": "gpt-5.6-terra",
                "usage": usage,
            }
        )

        result = KodaCore(processor).handle(InteractionRequest("user-01", "koda-01", "text", "Quem criou o Linux?"))

        self.assertEqual(result.provider, "chatgpt-plan")
        self.assertEqual(result.model, "gpt-5.6-terra")
        self.assertIs(result.usage, usage)

    def test_metadata_uses_processor_then_request_then_core_context(self) -> None:
        processor = StubProcessor(
            {
                "text": "Certo.",
                "interaction_state": "idle",
                "actions": [],
                "metadata": {"origin": "processor", "trace_id": "trace-01", "user_id": "ignored"},
            }
        )
        request = InteractionRequest(
            "user-01",
            "koda-01",
            "text",
            "Olá",
            metadata={"origin": "request", "channel": "test", "device_id": "ignored"},
        )

        result = KodaCore(processor).handle(request)

        self.assertEqual(
            result.metadata,
            {
                "origin": "request",
                "trace_id": "trace-01",
                "user_id": "user-01",
                "channel": "test",
                "device_id": "koda-01",
                "input_type": "text",
            },
        )

    def test_actions_are_copied_without_shared_mutable_state(self) -> None:
        response_actions = [{"type": "expression", "value": "attention"}]
        first = KodaCore(
            StubProcessor({"text": "", "interaction_state": "attention", "actions": response_actions})
        ).handle(InteractionRequest("user-01", "koda-01", "text", "Koda"))
        second = KodaCore(
            StubProcessor({"text": "", "interaction_state": "attention", "actions": response_actions})
        ).handle(InteractionRequest("user-02", "koda-02", "text", "Koda"))

        first.actions.append({"type": "display"})

        self.assertEqual(second.actions, [{"type": "expression", "value": "attention"}])
        self.assertEqual(response_actions, [{"type": "expression", "value": "attention"}])

    def test_processor_exceptions_propagate(self) -> None:
        request = InteractionRequest("user-01", "koda-01", "text", "Olá")

        with self.assertRaisesRegex(RuntimeError, "processor failure"):
            KodaCore(FailingProcessor()).handle(request)


if __name__ == "__main__":
    unittest.main()
