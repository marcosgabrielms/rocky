"""Testes do ciclo local de entrega de lembretes."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from services.reminder_scheduler import ReminderScheduler
from services.reminders import RECURRENCE_DAILY, RECURRENCE_WEEKLY, Reminder, ReminderRecurrence, ReminderService


class _RecordingDelivery:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.delivered: list[Reminder] = []

    def deliver(self, reminder: Reminder) -> bool:
        self.delivered.append(reminder)
        return self.result


class _FailingDelivery:
    def deliver(self, reminder: Reminder) -> bool:
        del reminder
        raise RuntimeError("canal indisponivel")


class ReminderSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.database_path = Path(self._temporary_directory.name) / "reminders.sqlite3"

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _service(self) -> ReminderService:
        return ReminderService(self.database_path)

    def test_due_reminder_is_delivered_only_after_the_channel_confirms(self) -> None:
        service = self._service()
        reminder = service.create("user-a", "beber agua", 10.0, now=1.0)
        delivery = _RecordingDelivery()
        scheduler = ReminderScheduler(service, delivery, clock=lambda: 10.0)

        self.assertEqual(scheduler.poll(), (reminder,))
        self.assertEqual(delivery.delivered, [reminder])
        self.assertEqual(service.get_due(10.0), ())
        self.assertEqual(service.list_pending("user-a"), ())
        service.close()

    def test_failed_or_unavailable_delivery_keeps_reminder_pending(self) -> None:
        service = self._service()
        reminder = service.create("user-a", "beber agua", 10.0, now=1.0)

        failed_delivery = _RecordingDelivery(result=False)
        failed_scheduler = ReminderScheduler(service, failed_delivery, clock=lambda: 10.0)
        error_scheduler = ReminderScheduler(service, _FailingDelivery(), clock=lambda: 10.0)
        unavailable_scheduler = ReminderScheduler(service, clock=lambda: 10.0)

        self.assertEqual(failed_scheduler.poll(), ())
        self.assertEqual(failed_delivery.delivered, [reminder])
        self.assertEqual(service.get_due(10.0), (reminder,))
        self.assertEqual(error_scheduler.poll(), ())
        self.assertEqual(service.get_due(10.0), (reminder,))
        self.assertEqual(unavailable_scheduler.poll(), ())
        self.assertEqual(service.get_due(10.0), (reminder,))
        service.close()

    def test_successful_delivery_is_not_repeated(self) -> None:
        service = self._service()
        reminder = service.create("user-a", "beber agua", 10.0, now=1.0)
        delivery = _RecordingDelivery()
        scheduler = ReminderScheduler(service, delivery, clock=lambda: 10.0)

        self.assertEqual(scheduler.poll(), (reminder,))
        self.assertEqual(scheduler.poll(), ())
        self.assertEqual(delivery.delivered, [reminder])
        service.close()

    def test_restart_finds_overdue_pending_but_not_delivered_or_cancelled_reminders(self) -> None:
        first_service = self._service()
        pending = first_service.create("user-a", "pendente", 10.0, now=1.0)
        delivered = first_service.create("user-a", "entregue", 10.0, now=1.0)
        cancelled = first_service.create("user-a", "cancelado", 10.0, now=1.0)
        self.assertTrue(first_service.mark_delivered(delivered.id, now=2.0))
        self.assertTrue(first_service.cancel("user-a", cancelled.id, now=2.0))
        first_service.close()

        second_service = self._service()
        delivery = _RecordingDelivery()
        scheduler = ReminderScheduler(second_service, delivery, clock=lambda: 10.0)

        self.assertEqual(scheduler.poll(), (pending,))
        self.assertEqual(delivery.delivered, [pending])
        second_service.close()

    def test_daily_and_weekly_delivery_rearm_one_next_pending_occurrence(self) -> None:
        service = self._service()
        daily = service.create(
            "user-a", "beber agua", 100.0,
            recurrence=ReminderRecurrence(RECURRENCE_DAILY), now=1.0,
        )
        weekly = service.create(
            "user-a", "revisar agenda", 100.0,
            recurrence=ReminderRecurrence(RECURRENCE_WEEKLY, 0), now=1.0,
        )
        delivery = _RecordingDelivery()
        scheduler = ReminderScheduler(service, delivery, clock=lambda: 100.0)

        self.assertEqual(scheduler.poll(), (daily, weekly))
        pending = service.list_pending("user-a")
        self.assertEqual(tuple(item.id for item in pending), (daily.id, weekly.id))
        self.assertEqual(tuple(item.due_at for item in pending), (86500.0, 604900.0))
        self.assertEqual(tuple(item.recurrence_kind for item in pending), (RECURRENCE_DAILY, RECURRENCE_WEEKLY))
        self.assertEqual(scheduler.poll(), ())
        service.close()

    def test_recurring_schedule_survives_restart_and_cancelled_recurrence_does_not_fire(self) -> None:
        first_service = self._service()
        recurring = first_service.create(
            "user-a", "beber agua", 100.0,
            recurrence=ReminderRecurrence(RECURRENCE_DAILY), now=1.0,
        )
        cancelled = first_service.create(
            "user-a", "nao entregar", 100.0,
            recurrence=ReminderRecurrence(RECURRENCE_WEEKLY, 0), now=1.0,
        )
        self.assertTrue(first_service.cancel("user-a", cancelled.id, now=2.0))
        first_service.close()

        second_service = self._service()
        delivery = _RecordingDelivery()
        scheduler = ReminderScheduler(second_service, delivery, clock=lambda: 100.0)
        self.assertEqual(scheduler.poll(), (recurring,))
        second_service.close()

        third_service = self._service()
        restarted_delivery = _RecordingDelivery()
        restarted_scheduler = ReminderScheduler(third_service, restarted_delivery, clock=lambda: 100.0)
        self.assertEqual(restarted_scheduler.poll(), ())
        self.assertEqual(restarted_delivery.delivered, [])
        self.assertEqual(third_service.list_pending("user-a")[0].due_at, 86500.0)
        third_service.close()

    def test_snooze_rechedules_only_its_pending_reminder_and_preserves_user_isolation(self) -> None:
        service = self._service()
        reminder = service.create("user-a", "fazer atividade", 10.0, now=1.0)
        other_user = service.create("user-b", "fazer atividade", 10.0, now=1.0)

        snoozed = service.snooze("user-a", reminder.id, 900.0, now=100.0)

        self.assertIsNotNone(snoozed)
        self.assertEqual(snoozed.due_at, 1000.0)
        self.assertIsNone(service.snooze("user-a", other_user.id, 60.0, now=100.0))
        self.assertEqual(service.get_due(999.0), (other_user,))
        self.assertEqual(service.get_due(1000.0), (other_user, snoozed))
        service.close()

    def test_worker_has_an_explicit_start_and_stop_lifecycle(self) -> None:
        service = self._service()
        scheduler = ReminderScheduler(service, interval_seconds=60.0)

        scheduler.start()
        self.assertTrue(scheduler.is_running)
        scheduler.start()
        scheduler.stop()
        self.assertFalse(scheduler.is_running)
        service.close()


if __name__ == "__main__":
    unittest.main()
