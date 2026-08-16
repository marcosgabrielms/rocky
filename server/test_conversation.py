"""Testes determinísticos para wake word e contexto curto do Rocky."""

import unittest
from collections.abc import Sequence
from unittest.mock import patch

from services.conversation import CONTEXT_TTL_SECONDS, MAX_CONTEXT_MESSAGES, ConversationManager
from services.llm import ConversationMessage, LLMResult, OpenRouterLLMClient


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        self.calls.append((message, tuple(context or ())))
        return LLMResult(text=f"Resposta para: {message}")


class ConversationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.llm_client = FakeLLMClient()
        self.conversation = ConversationManager(10000, self.llm_client, clock=self.clock)

    def wake(self, device_id: str = "rocky-01", text: str = "Ok, Rocky") -> dict[str, object]:
        return self.conversation.build_response(text, device_id)

    def ask(self, text: str, device_id: str = "rocky-01") -> dict[str, object]:
        self.wake(device_id)
        return self.conversation.build_response(text, device_id)

    def assert_wake(self, text: str) -> None:
        response = self.wake(text=text)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [{"type": "expression", "value": "attention"}])

    def assert_not_wake(self, text: str) -> None:
        response = self.conversation.build_response(text, "idle-test")

        self.assertEqual(response["interaction_state"], "idle")
        self.assertEqual(response["actions"], [])

    def test_accepts_ok_rocky_with_punctuation_and_spacing_variants(self) -> None:
        for text in ("Ok, Rocky", "Ok Rocky", "Ok, Rocky!", "ok rocky"):
            with self.subTest(text=text):
                self.assert_wake(text)

    def test_accepts_rocky_as_temporary_fallback(self) -> None:
        self.assert_wake("Rocky")

    def test_rejects_ok_without_rocky(self) -> None:
        self.assert_not_wake("Ok")

    def test_rejects_common_phrase_with_rocky(self) -> None:
        self.assert_not_wake("Onde está Rocky")

    def test_second_question_receives_context_from_first_question(self) -> None:
        self.ask("Quem criou o Linux?")
        self.clock.now += 1
        self.ask("E de onde ele é?")

        self.assertEqual(
            self.llm_client.calls[-1][1],
            (
                ConversationMessage(role="user", content="Quem criou o Linux?"),
                ConversationMessage(role="assistant", content="Resposta para: Quem criou o Linux?"),
            ),
        )

    def test_context_is_separated_by_device_id(self) -> None:
        self.ask("Pergunta do primeiro Rocky", "rocky-01")
        self.clock.now += 1
        self.ask("Pergunta do segundo Rocky", "rocky-02")

        self.assertEqual(self.llm_client.calls[-1][1], ())

    def test_context_keeps_at_most_six_messages(self) -> None:
        for index in range(4):
            self.ask(f"Pergunta {index}")
            self.clock.now += 1

        history = self.conversation._histories["rocky-01"].messages
        self.assertEqual(len(history), MAX_CONTEXT_MESSAGES)
        self.assertEqual(history[0].content, "Pergunta 1")
        self.assertEqual(history[-1].content, "Resposta para: Pergunta 3")

    def test_context_expires_after_five_minutes_without_valid_interaction(self) -> None:
        self.ask("Primeira pergunta")
        self.clock.now += CONTEXT_TTL_SECONDS + 1
        self.ask("Pergunta após expiração")

        self.assertEqual(self.llm_client.calls[-1][1], ())

    def test_wake_does_not_enter_history(self) -> None:
        self.wake()

        self.assertNotIn("rocky-01", self.conversation._histories)

    def test_hours_does_not_enter_history(self) -> None:
        self.wake()
        with patch("services.conversation.get_current_time", return_value="12:00"):
            self.conversation.build_response("horas", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_question_in_idle_does_not_enter_history(self) -> None:
        self.conversation.build_response("Quem criou o Linux?", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_empty_text_does_not_enter_history(self) -> None:
        self.wake()
        self.conversation.build_response("", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_openrouter_preserves_context_roles(self) -> None:
        messages = OpenRouterLLMClient._build_messages(
            "Nova pergunta",
            (
                ConversationMessage(role="user", content="Pergunta anterior"),
                ConversationMessage(role="assistant", content="Resposta anterior"),
            ),
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])


if __name__ == "__main__":
    unittest.main()
