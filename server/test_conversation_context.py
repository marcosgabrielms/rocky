"""Testes do armazenamento em memÃ³ria do contexto de conversa."""

import unittest

from services.conversation import ConversationManager, InteractionMode
from services.conversation_context import ConversationContextStore
from services.llm.base import LLMResult


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _LLMDouble:
    def ask(self, message, context=None) -> LLMResult:
        del context
        return LLMResult(text=f"Resposta para: {message}")


class ConversationContextStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.store = ConversationContextStore(max_messages=2, ttl_seconds=10)

    def test_get_add_and_clear_are_isolated_by_user_id(self) -> None:
        self.store.add_exchange("user-a", "Pergunta A", "Resposta A", self.clock())
        self.store.add_exchange("user-b", "Pergunta B", "Resposta B", self.clock())

        self.assertEqual(self.store.get_context("user-a", self.clock())[0].content, "Pergunta A")
        self.assertEqual(self.store.get_context("user-b", self.clock())[0].content, "Pergunta B")
        self.store.clear("user-a")
        self.assertEqual(self.store.get_context("user-a", self.clock()), ())
        self.assertEqual(self.store.get_context("user-b", self.clock())[0].content, "Pergunta B")

    def test_ttl_cleanup_and_message_limit_preserve_existing_semantics(self) -> None:
        self.store.add_exchange("user", "Primeira", "Resposta", self.clock())
        self.assertEqual(len(self.store.get_context("user", self.clock())), 2)
        self.clock.now = 10.0
        self.assertEqual(self.store.get_context("user", self.clock()), ())
        self.assertNotIn("user", self.store._histories)

    def test_manager_delegates_context_storage_without_parallel_history(self) -> None:
        manager = ConversationManager(0, _LLMDouble(), clock=self.clock, context_store=self.store)
        manager.build_response(
            "Pergunta", "robot", user_id="local-user", interaction_mode=InteractionMode.COMMAND
        )

        self.assertFalse(hasattr(manager, "_histories"))
        self.assertIn("local-user", self.store._histories)


if __name__ == "__main__":
    unittest.main()
