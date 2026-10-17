"""Testes deterministas dos lembretes locais persistentes."""

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zoneinfo import ZoneInfo

from services.conversation import ConversationManager, InteractionMode
from services.conversation_context import ConversationContextStore
from services.llm.base import LLMResult
from services.long_term_memory import LongTermMemoryStore
from services.reminders import (
    RECURRENCE_DAILY,
    RECURRENCE_WEEKLY,
    ReminderRecurrence,
    ReminderService,
    parse_reminder_request,
)


SAO_PAULO = ZoneInfo("America/Sao_Paulo")
NOW = datetime(2026, 10, 16, 10, 0, tzinfo=SAO_PAULO)


class _LLMDouble:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ask(self, message: str, context=None) -> LLMResult:
        del context
        self.calls.append(message)
        return LLMResult(text=f"Resposta para: {message}")


class ReminderServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "reminders.sqlite3"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_parser_accepts_relative_minutes_hours_and_today_time(self) -> None:
        minutes = parse_reminder_request("me lembra daqui a 2 minutos de beber agua", now=NOW)
        hours = parse_reminder_request("me lembra daqui a 2 horas de fazer a atividade", now=NOW)
        today = parse_reminder_request("me lembra hoje as 18:30 de ir ao supermercado", now=NOW)
        today_hour = parse_reminder_request("me lembra hoje 18 de estudar", now=NOW)

        self.assertEqual(minutes.text, "beber agua")
        self.assertEqual(minutes.due_at, datetime(2026, 10, 16, 10, 2, tzinfo=SAO_PAULO).timestamp())
        self.assertEqual(hours.due_at, datetime(2026, 10, 16, 12, 0, tzinfo=SAO_PAULO).timestamp())
        self.assertEqual(today.due_at, datetime(2026, 10, 16, 18, 30, tzinfo=SAO_PAULO).timestamp())
        self.assertEqual(today_hour.due_at, datetime(2026, 10, 16, 18, 0, tzinfo=SAO_PAULO).timestamp())
        self.assertIsNone(parse_reminder_request("me lembra hoje as 09:00 de algo", now=NOW))

    def test_parser_accepts_daily_and_weekly_recurrence(self) -> None:
        daily = parse_reminder_request("me lembra todo dia as 18:30 de beber agua", now=NOW)
        next_daily = parse_reminder_request("me lembra todo dia as 09:00 de beber agua", now=NOW)
        weekly = parse_reminder_request("me lembra toda segunda as 08:00 de revisar agenda", now=NOW)

        self.assertEqual(daily.recurrence, ReminderRecurrence(RECURRENCE_DAILY))
        self.assertEqual(daily.due_at, datetime(2026, 10, 16, 18, 30, tzinfo=SAO_PAULO).timestamp())
        self.assertEqual(next_daily.due_at, datetime(2026, 10, 17, 9, 0, tzinfo=SAO_PAULO).timestamp())
        self.assertEqual(weekly.recurrence, ReminderRecurrence(RECURRENCE_WEEKLY, 0))
        self.assertEqual(weekly.due_at, datetime(2026, 10, 19, 8, 0, tzinfo=SAO_PAULO).timestamp())

    def test_persists_across_restart_and_isolates_users(self) -> None:
        first_service = ReminderService(self.database_path)
        first_service.create("user-a", "fazer atividade", 200.0, now=1.0)
        first_service.create("user-b", "comprar cafe", 300.0, now=2.0)
        first_service.close()

        second_service = ReminderService(self.database_path)
        self.assertEqual(tuple(item.text for item in second_service.list_pending("user-a")), ("fazer atividade",))
        self.assertEqual(tuple(item.text for item in second_service.list_pending("user-b")), ("comprar cafe",))
        second_service.close()

    def test_pending_cancel_due_and_status_transitions(self) -> None:
        service = ReminderService(self.database_path)
        due = service.create("user-a", "ir ao supermercado", 100.0, now=1.0)
        later = service.create("user-a", "fazer atividade", 200.0, now=2.0)
        service.create("user-b", "outro usuario", 50.0, now=3.0)

        self.assertEqual(due.status, "pending")
        self.assertEqual(tuple(item.id for item in service.get_due(100.0)), (3, due.id))
        self.assertTrue(service.mark_delivered(due.id, now=101.0))
        self.assertTrue(service.mark_acknowledged(due.id, now=102.0))
        self.assertTrue(service.cancel("user-a", later.id, now=103.0))
        self.assertFalse(service.cancel("user-b", later.id, now=104.0))
        self.assertEqual(service.list_pending("user-a"), ())
        service.close()

    def test_safe_match_requires_exactly_one_pending_reminder(self) -> None:
        service = ReminderService(self.database_path)
        supermarket = service.create("user-a", "ir ao supermercado", 100.0, now=1.0)
        service.create("user-a", "voltar ao supermercado", 200.0, now=2.0)
        service.create("user-b", "ir ao supermercado", 100.0, now=3.0)

        self.assertEqual(service.find_pending_matches("user-a", "supermercado"), (supermarket, service.list_pending("user-a")[1]))
        self.assertEqual(service.find_pending_matches("user-a", "ir ao supermercado"), (supermarket,))
        self.assertEqual(service.find_pending_matches("user-b", "supermercado"), (service.list_pending("user-b")[0],))
        service.close()

    def test_reschedule_and_snooze_are_scoped_to_a_pending_reminder_of_the_same_user(self) -> None:
        service = ReminderService(self.database_path)
        reminder = service.create("user-a", "ir ao mercado", 100.0, now=1.0)
        other_user = service.create("user-b", "ir ao mercado", 100.0, now=1.0)

        rescheduled = service.reschedule("user-a", reminder.id, 300.0, now=2.0)
        snoozed = service.snooze("user-a", reminder.id, 60.0, now=400.0)

        self.assertEqual(rescheduled.due_at, 300.0)
        self.assertEqual(snoozed.due_at, 460.0)
        self.assertIsNone(service.reschedule("user-a", other_user.id, 500.0, now=2.0))
        self.assertTrue(service.mark_delivered(reminder.id, now=461.0))
        self.assertIsNone(service.reschedule("user-a", reminder.id, 600.0, now=462.0))
        service.close()

    def test_conversation_handles_reminders_without_memory_or_llm(self) -> None:
        reminder_service = ReminderService(self.database_path)
        memory_store = LongTermMemoryStore(Path(self._temporary_directory.name) / "memory.sqlite3")
        context_store = ConversationContextStore()
        llm = _LLMDouble()
        manager = ConversationManager(
            0,
            llm,
            context_store=context_store,
            memory_store=memory_store,
            reminder_service=reminder_service,
            reminder_now=lambda: NOW,
        )

        created = manager.build_response(
            "me lembra daqui a 2 horas de fazer a atividade",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        listed = manager.build_response(
            "quais lembretes eu tenho",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        cancelled = manager.build_response(
            "cancela o lembrete da atividade",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )
        manager.build_response(
            "lembra que eu prefiro respostas curtas",
            "robot",
            user_id="local-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertIn("Lembrete salvo", created["text"])
        self.assertIn("fazer a atividade", listed["text"])
        self.assertEqual(cancelled["text"], "Lembrete cancelado.")
        self.assertEqual(reminder_service.list_pending("local-user"), ())
        self.assertEqual(context_store.get_context("local-user", now=0.0), ())
        self.assertEqual(memory_store.list("local-user")[0].content, "eu prefiro respostas curtas")
        self.assertEqual(llm.calls, [])
        reminder_service.close()
        memory_store.close()

    def test_conversation_rechedules_delays_and_rejects_ambiguous_reminders_without_llm(self) -> None:
        reminder_service = ReminderService(self.database_path)
        manager = ConversationManager(
            0,
            _LLMDouble(),
            reminder_service=reminder_service,
            reminder_now=lambda: NOW,
        )
        reminder_service.create("local-user", "ir ao mercado", NOW.timestamp() + 3600, now=NOW.timestamp())

        rescheduled = manager.build_response(
            "muda o lembrete do mercado para 19 horas",
            "robot", user_id="local-user", interaction_mode=InteractionMode.COMMAND,
        )
        delayed = manager.build_response(
            "adia o lembrete do mercado por 30 minutos",
            "robot", user_id="local-user", interaction_mode=InteractionMode.COMMAND,
        )
        reminder_service.create("local-user", "comprar mercado", NOW.timestamp() + 3600, now=NOW.timestamp())
        ambiguous = manager.build_response(
            "muda o lembrete do mercado para 20 horas",
            "robot", user_id="local-user", interaction_mode=InteractionMode.COMMAND,
        )

        self.assertIn("reagendado", rescheduled["text"])
        self.assertIn("adiado", delayed["text"])
        self.assertIn("mais de um", ambiguous["text"])
        self.assertEqual(
            tuple(reminder.due_at for reminder in reminder_service.list_pending("local-user")),
            (NOW.timestamp() + 1800, NOW.timestamp() + 3600),
        )
        reminder_service.close()

    def test_conversation_snoozes_only_one_delivered_reminder(self) -> None:
        reminder_service = ReminderService(self.database_path)
        manager = ConversationManager(
            0,
            _LLMDouble(),
            reminder_service=reminder_service,
            reminder_now=lambda: NOW,
        )
        reminder = reminder_service.create("local-user", "tomar remedio", NOW.timestamp(), now=NOW.timestamp())
        self.assertTrue(reminder_service.mark_delivered(reminder.id, now=NOW.timestamp()))

        snoozed = manager.build_response(
            "me lembra de novo daqui a 15 minutos",
            "robot", user_id="local-user", interaction_mode=InteractionMode.COMMAND,
        )
        not_associated = manager.build_response(
            "me lembra de novo daqui a 15 minutos",
            "robot", user_id="local-user", interaction_mode=InteractionMode.COMMAND,
        )

        self.assertIn("adiado", snoozed["text"])
        self.assertIn("Nao tenho", not_associated["text"])
        pending = reminder_service.list_pending("local-user")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].due_at, NOW.timestamp() + 900)
        reminder_service.close()


if __name__ == "__main__":
    unittest.main()
