"""Testes determinísticos para a wake word do Rocky."""

import unittest

from services.conversation import ConversationManager
from services.llm import LLMResult


class FakeLLMClient:
    def ask(self, _message: str, _context: list[str] | None = None) -> LLMResult:
        return LLMResult(text="Resposta simulada.")


class WakeWordTest(unittest.TestCase):
    def build_response(self, text: str) -> dict[str, object]:
        conversation = ConversationManager(10000, FakeLLMClient())
        return conversation.build_response(text, "wake-test")

    def assert_wake(self, text: str) -> None:
        response = self.build_response(text)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [{"type": "expression", "value": "attention"}])

    def assert_not_wake(self, text: str) -> None:
        response = self.build_response(text)

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


if __name__ == "__main__":
    unittest.main()
