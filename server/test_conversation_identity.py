"""Testes da separaÃ§Ã£o entre identidade de conversa e dispositivo."""

import unittest

from koda_core import ConversationCoreAdapter
from services.conversation import ConversationManager, InteractionMode
from services.interaction import KodaInteractionService
from services.llm.base import LLMResult


class _LLMDouble:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def ask(self, message: str, context=None) -> LLMResult:
        self.calls.append((message, tuple(context or ())))
        return LLMResult(text=f"Resposta para: {message}")


class ConversationIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.llm = _LLMDouble()
        self.manager = ConversationManager(0, self.llm)
        self.service = KodaInteractionService(ConversationCoreAdapter(self.manager))

    def ask(self, user_id: str, device_id: str, text: str):
        decision = self.service.classify(
            user_id=user_id,
            device_id=device_id,
            input_type="text",
            text=text,
            interaction_mode=InteractionMode.COMMAND,
        )
        self.service.build_response(decision)
        return decision

    def test_same_user_keeps_context_across_devices_without_mixing_device_identity(self) -> None:
        first = self.ask("local-user", "robot", "Quem criou o Linux?")
        second = self.ask("local-user", "android", "E de onde ele Ã©?")

        self.assertEqual(first.user_id, "local-user")
        self.assertEqual(first.device_id, "robot")
        self.assertEqual(second.user_id, "local-user")
        self.assertEqual(second.device_id, "android")
        self.assertEqual(
            tuple((message.role, message.content) for message in self.llm.calls[1][1]),
            (
                ("user", "Quem criou o Linux?"),
                ("assistant", "Resposta para: Quem criou o Linux?"),
            ),
        )
        self.assertEqual(set(self.manager.context_store._histories), {"local-user"})

    def test_different_users_keep_separate_contexts(self) -> None:
        self.ask("user-a", "robot", "Pergunta do usuario A")
        self.ask("user-b", "android", "Pergunta do usuario B")

        self.assertEqual(self.llm.calls[1][1], ())
        self.assertEqual(set(self.manager.context_store._histories), {"user-a", "user-b"})

    def test_legacy_calls_without_user_id_remain_scoped_to_the_device(self) -> None:
        self.manager.build_response("Pergunta legado", "robot", interaction_mode=InteractionMode.COMMAND)

        self.assertIn("robot", self.manager.context_store._histories)


if __name__ == "__main__":
    unittest.main()
