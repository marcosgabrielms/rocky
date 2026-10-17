"""Testes do armazenamento SQLite do contexto curto de conversa."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.conversation import ConversationManager, InteractionMode
from services.conversation_context import (
    CONTEXT_TTL_SECONDS,
    DEFAULT_RECENT_CONTEXT_POLICY,
    MAX_CONTEXT_MESSAGES,
    RecentContextPolicy,
    SqliteConversationContextStore,
)
from services.llm.base import LLMResult


class _LLMDouble:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def ask(self, message: str, context=None) -> LLMResult:
        self.calls.append((message, tuple(context or ())))
        return LLMResult(text=f"Resposta para: {message}")


class SqliteConversationContextStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "context.sqlite3"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def create_store(
        self,
        policy: RecentContextPolicy | None = None,
    ) -> SqliteConversationContextStore:
        return SqliteConversationContextStore(
            self.database_path,
            policy=policy or RecentContextPolicy(max_messages=6, ttl_seconds=10),
        )

    def test_default_policy_preserves_the_current_recent_context_limits(self) -> None:
        store = SqliteConversationContextStore(self.database_path)

        self.assertEqual(DEFAULT_RECENT_CONTEXT_POLICY.max_messages, MAX_CONTEXT_MESSAGES)
        self.assertEqual(DEFAULT_RECENT_CONTEXT_POLICY.ttl_seconds, CONTEXT_TTL_SECONDS)
        self.assertEqual(store.policy, DEFAULT_RECENT_CONTEXT_POLICY)
        store.close()

    def test_policy_is_configurable_for_tests(self) -> None:
        store = self.create_store(RecentContextPolicy(max_messages=2, ttl_seconds=5))
        store.add_exchange("local-user", "Pergunta 1", "Resposta 1", now=1.0)
        store.add_exchange("local-user", "Pergunta 2", "Resposta 2", now=2.0)

        self.assertEqual(
            tuple(message.content for message in store.get_context("local-user", now=3.0)),
            ("Pergunta 2", "Resposta 2"),
        )
        self.assertEqual(store.get_context("local-user", now=7.0), ())
        store.close()

    def test_context_survives_store_recreation_with_original_order(self) -> None:
        first_store = self.create_store()
        first_store.add_exchange("local-user", "Pergunta 1", "Resposta 1", now=1.0)
        first_store.add_exchange("local-user", "Pergunta 2", "Resposta 2", now=2.0)
        first_store.close()

        second_store = self.create_store()
        self.assertEqual(
            tuple((message.role, message.content) for message in second_store.get_context("local-user", now=3.0)),
            (
                ("user", "Pergunta 1"),
                ("assistant", "Resposta 1"),
                ("user", "Pergunta 2"),
                ("assistant", "Resposta 2"),
            ),
        )
        second_store.close()

    def test_same_user_shares_context_across_devices_and_users_stay_isolated(self) -> None:
        store = self.create_store()
        llm = _LLMDouble()
        manager = ConversationManager(0, llm, clock=lambda: 1.0, context_store=store)

        manager.build_response(
            "Quem criou o Linux?",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        manager.build_response(
            "E de onde ele e?",
            "android",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        manager.build_response(
            "Pergunta de outra pessoa",
            "robot",
            user_id="other-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertEqual(
            tuple((message.role, message.content) for message in llm.calls[1][1]),
            (
                ("user", "Quem criou o Linux?"),
                ("assistant", "Resposta para: Quem criou o Linux?"),
            ),
        )
        self.assertEqual(llm.calls[2][1], ())
        store.close()

    def test_ttl_removes_expired_context_before_and_after_reopen(self) -> None:
        first_store = self.create_store()
        first_store.add_exchange("local-user", "Pergunta", "Resposta", now=1.0)
        first_store.close()

        second_store = self.create_store()
        self.assertEqual(second_store.get_context("local-user", now=11.0), ())
        second_store.close()

        third_store = self.create_store()
        self.assertEqual(third_store.get_context("local-user", now=12.0), ())
        third_store.close()

    def test_message_limit_keeps_only_the_six_most_recent_messages(self) -> None:
        store = self.create_store()
        for index in range(4):
            store.add_exchange("local-user", f"Pergunta {index}", f"Resposta {index}", now=float(index))

        self.assertEqual(
            tuple(message.content for message in store.get_context("local-user", now=4.0)),
            (
                "Pergunta 1",
                "Resposta 1",
                "Pergunta 2",
                "Resposta 2",
                "Pergunta 3",
                "Resposta 3",
            ),
        )
        store.close()

    def test_clear_removes_persisted_context(self) -> None:
        first_store = self.create_store()
        first_store.add_exchange("local-user", "Pergunta", "Resposta", now=1.0)
        first_store.clear("local-user")
        first_store.close()

        second_store = self.create_store()
        self.assertEqual(second_store.get_context("local-user", now=2.0), ())
        second_store.close()

    def test_normal_operations_purge_expired_users_that_do_not_return(self) -> None:
        store = self.create_store()
        store.add_exchange("expired-user", "Pergunta", "Resposta", now=1.0)
        store.add_exchange("active-user", "Pergunta", "Resposta", now=9.0)

        self.assertEqual(len(store.get_context("active-user", now=11.0)), 2)
        self.assertEqual(store.get_context("expired-user", now=11.0), ())
        store.close()


if __name__ == "__main__":
    unittest.main()
