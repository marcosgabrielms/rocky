"""Agendamento local de lembretes sem acoplamento ao canal de entrega."""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock, Thread, current_thread
import time
from typing import Protocol

from services.reminders import Reminder, ReminderServiceProtocol


class ReminderDelivery(Protocol):
    """Canal futuro que confirma se conseguiu entregar um lembrete."""

    def deliver(self, reminder: Reminder) -> bool: ...


class UnavailableReminderDelivery:
    """Canal padrao seguro enquanto nao existe destino proativo conectado."""

    def deliver(self, reminder: Reminder) -> bool:
        del reminder
        return False


class ReminderScheduler:
    """Consulta vencidos e so marca entrega depois de confirmacao do canal."""

    def __init__(
        self,
        reminder_service: ReminderServiceProtocol,
        delivery: ReminderDelivery | None = None,
        *,
        clock: Callable[[], float] = time.time,
        interval_seconds: float = 30.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds deve ser maior que zero")
        self._reminder_service = reminder_service
        self._delivery = delivery or UnavailableReminderDelivery()
        self._clock = clock
        self._interval_seconds = interval_seconds
        self._stop_event = Event()
        self._poll_lock = Lock()
        self._thread: Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def poll(self) -> tuple[Reminder, ...]:
        """Executa um ciclo sincrono; util para testes e para o worker controlado."""
        if not self._poll_lock.acquire(blocking=False):
            return ()
        try:
            now = self._clock()
            delivered: list[Reminder] = []
            for reminder in self._reminder_service.get_due(now):
                try:
                    accepted = self._delivery.deliver(reminder)
                except Exception:
                    accepted = False
                if accepted and self._reminder_service.mark_delivered(reminder.id, now=now):
                    delivered.append(reminder)
            return tuple(delivered)
        finally:
            self._poll_lock.release()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="koda-reminder-scheduler", daemon=True)
        self._thread.start()

    def stop(self, *, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is None or thread is current_thread():
            return
        thread.join(timeout_seconds)
        if not thread.is_alive():
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.poll()
            self._stop_event.wait(self._interval_seconds)
