"""Testes da composição compartilhada do runtime do Koda."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import app
from fastapi.testclient import TestClient
from runtime import KodaRuntime
from koda_core import ActionPolicy, PendingActionAuthorizations
from services.action_authorization import ActionAuthorizationSession
from services.command_session import CommandSession
from services.conversation import InteractionMode
from services.conversation_context import RecentContextPolicy, SqliteConversationContextStore
from services.llm.base import LLMResult
from services.long_term_memory import LongTermMemoryStore
from services.reminders import ReminderService
from services.reminder_scheduler import ReminderScheduler


class _LLMDouble:
    def ask(self, *_args, **_kwargs):
        raise AssertionError("O LLM não deve ser chamado ao compor o runtime.")


class _ContextLLM:
    def ask(self, message, context=None):
        del context
        return LLMResult(text=f"Resposta para: {message}")


class KodaRuntimeTest(unittest.TestCase):
    def test_create_builds_one_shared_graph(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            runtime = KodaRuntime.create(
                _LLMDouble(),
                context_database_path=Path(temporary_directory) / "context.sqlite3",
                memory_database_path=Path(temporary_directory) / "memory.sqlite3",
                reminder_database_path=Path(temporary_directory) / "reminders.sqlite3",
            )

            self.assertIs(runtime.conversation_core_adapter._conversation_manager, runtime.conversation_manager)
            self.assertIsInstance(runtime.command_session, CommandSession)
            self.assertIsInstance(runtime.conversation_context_store, SqliteConversationContextStore)
            self.assertIsInstance(runtime.long_term_memory_store, LongTermMemoryStore)
            self.assertIsInstance(runtime.reminder_service, ReminderService)
            self.assertIsInstance(runtime.reminder_scheduler, ReminderScheduler)
            self.assertIsInstance(runtime.action_policy, ActionPolicy)
            self.assertIsInstance(runtime.pending_action_authorizations, PendingActionAuthorizations)
            self.assertIsInstance(runtime.action_authorization_session, ActionAuthorizationSession)
            self.assertIs(runtime.interaction_service._conversation_core_adapter, runtime.conversation_core_adapter)
            self.assertIs(runtime.conversation_manager.context_store, runtime.conversation_context_store)
            self.assertIs(runtime.conversation_manager.memory_store, runtime.long_term_memory_store)
            self.assertIs(runtime.conversation_manager.reminder_service, runtime.reminder_service)
            self.assertIs(
                runtime.conversation_manager.action_authorization_session,
                runtime.action_authorization_session,
            )
            self.assertIs(runtime.action_authorization_session._policy, runtime.action_policy)
            self.assertIs(
                runtime.action_authorization_session._pending_authorizations,
                runtime.pending_action_authorizations,
            )
            runtime.conversation_context_store.close()
            runtime.long_term_memory_store.close()
            runtime.reminder_service.close()

    def test_app_consumes_the_composed_shared_instances(self) -> None:
        self.assertIs(app.conversation_manager, app.runtime.conversation_manager)
        self.assertIs(app.conversation_core_adapter, app.runtime.conversation_core_adapter)
        self.assertIs(app.command_session, app.runtime.command_session)
        self.assertIs(app.interaction_service, app.runtime.interaction_service)

    def test_command_session_state_is_shared_for_the_app_lifetime(self) -> None:
        device_id = "runtime-session-test"
        app.command_session.close(device_id)
        app.command_session.open(device_id)
        self.assertTrue(app.runtime.command_session.is_active(device_id))
        app.command_session.close(device_id)

    def test_fastapi_lifecycle_starts_and_stops_the_reminder_scheduler(self) -> None:
        app.runtime.reminder_scheduler.stop()

        with TestClient(app.app) as client:
            self.assertTrue(app.runtime.reminder_scheduler.is_running)
            self.assertEqual(client.get("/health").status_code, 200)

        self.assertFalse(app.runtime.reminder_scheduler.is_running)

    def test_context_is_restored_when_runtime_is_recreated(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "context.sqlite3"
            first_runtime = KodaRuntime.create(
                _ContextLLM(),
                context_database_path=database_path,
                memory_database_path=Path(temporary_directory) / "memory.sqlite3",
                reminder_database_path=Path(temporary_directory) / "reminders.sqlite3",
            )
            first_runtime.conversation_manager.build_response(
                "Quem criou o Linux?",
                "robot",
                user_id="local-user",
                interaction_mode=InteractionMode.COMMAND,
            )
            first_runtime.conversation_context_store.close()
            first_runtime.long_term_memory_store.close()
            first_runtime.reminder_service.close()

            second_runtime = KodaRuntime.create(
                _ContextLLM(),
                context_database_path=database_path,
                memory_database_path=Path(temporary_directory) / "memory.sqlite3",
                reminder_database_path=Path(temporary_directory) / "reminders.sqlite3",
            )
            decision = second_runtime.conversation_manager.classify(
                "E de onde ele e?",
                "android",
                user_id="local-user",
                interaction_mode=InteractionMode.COMMAND,
            )

            self.assertEqual(
                tuple((message.role, message.content) for message in decision.context),
                (
                    ("user", "Quem criou o Linux?"),
                    ("assistant", "Resposta para: Quem criou o Linux?"),
                ),
            )
            second_runtime.conversation_context_store.close()
            second_runtime.long_term_memory_store.close()
            second_runtime.reminder_service.close()

    def test_runtime_uses_the_explicit_policy_and_purges_expired_context_on_startup(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "context.sqlite3"
            policy = RecentContextPolicy(max_messages=2, ttl_seconds=10)
            seeded_store = SqliteConversationContextStore(database_path, policy=policy)
            seeded_store.add_exchange("inactive-user", "Pergunta", "Resposta", now=1.0)
            seeded_store.close()

            runtime = KodaRuntime.create(
                _ContextLLM(),
                context_database_path=database_path,
                context_policy=policy,
                memory_database_path=Path(temporary_directory) / "memory.sqlite3",
                reminder_database_path=Path(temporary_directory) / "reminders.sqlite3",
                clock=lambda: 11.0,
            )

            self.assertIs(runtime.conversation_context_store.policy, policy)
            self.assertEqual(runtime.conversation_context_store.get_context("inactive-user", now=11.0), ())
            runtime.conversation_context_store.close()
            runtime.long_term_memory_store.close()
            runtime.reminder_service.close()


if __name__ == "__main__":
    unittest.main()
