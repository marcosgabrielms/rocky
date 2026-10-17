"""Lembretes locais persistentes, separados de memoria e contexto de conversa."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
import sqlite3
from threading import RLock
import time
from typing import Protocol
import unicodedata
from zoneinfo import ZoneInfo


REMINDER_STATUSES = frozenset({"pending", "delivered", "acknowledged", "cancelled"})
REMINDER_PREFIX_PATTERN = re.compile(r"^(?:me\s+lembra|lembra[-\s]me)\b", re.IGNORECASE)
RELATIVE_REMINDER_PATTERN = re.compile(
    r"^(?:me\s+lembra|lembra[-\s]me)\s+daqui\s+a\s+(?P<amount>\d+)\s+"
    r"(?P<unit>minuto(?:s)?|hora(?:s)?)\s+de\s+(?P<text>.+)$",
    re.IGNORECASE,
)
TODAY_REMINDER_PATTERN = re.compile(
    r"^(?:me\s+lembra|lembra[-\s]me)\s+hoje\s+(?:as\s*)?(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?\s+de\s+(?P<text>.+)$",
    re.IGNORECASE,
)
REMINDER_LIST_PATTERN = re.compile(r"^quais\s+lembretes\s+eu\s+tenho$", re.IGNORECASE)
REMINDER_CANCEL_PATTERN = re.compile(
    r"^cancela(?:r)?\s+(?:o\s+)?lembrete\s+(?:do|da|de)\s+(?P<text>.+)$",
    re.IGNORECASE,
)
REMINDER_RESCHEDULE_PATTERN = re.compile(
    r"^(?:muda|altera|reagenda)\s+(?:o\s+)?lembrete\s+(?:do|da|de)\s+(?P<text>.+?)\s+"
    r"para\s+(?P<hour>\d{1,2})(?:(?::(?P<minute>\d{2}))|\s+horas?)?$",
    re.IGNORECASE,
)
REMINDER_DELAY_PATTERN = re.compile(
    r"^adia(?:r)?\s+(?:o\s+)?lembrete\s+(?:do|da|de)\s+(?P<text>.+?)\s+por\s+"
    r"(?P<amount>\d+)\s+(?P<unit>minuto(?:s)?|hora(?:s)?)$",
    re.IGNORECASE,
)
REMINDER_SNOOZE_PATTERN = re.compile(
    r"^(?:me\s+lembra|lembra[-\s]me)\s+de\s+novo\s+daqui\s+a\s+(?P<amount>\d+)\s+"
    r"(?P<unit>minuto(?:s)?|hora(?:s)?)$",
    re.IGNORECASE,
)
DAILY_REMINDER_PATTERN = re.compile(
    r"^(?:me\s+lembra|lembra[-\s]me)\s+todo\s+dia\s+(?:as\s*)?(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?\s+de\s+(?P<text>.+)$",
    re.IGNORECASE,
)
WEEKLY_REMINDER_PATTERN = re.compile(
    r"^(?:me\s+lembra|lembra[-\s]me)\s+toda\s+(?P<weekday>segunda(?:\s+feira)?)\s+"
    r"(?:as\s*)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s+de\s+(?P<text>.+)$",
    re.IGNORECASE,
)
RECURRENCE_DAILY = "daily"
RECURRENCE_WEEKLY = "weekly"
REMINDER_RECURRENCES = frozenset({RECURRENCE_DAILY, RECURRENCE_WEEKLY})
WEEKDAY_BY_NAME = {"segunda": 0, "segunda feira": 0}
REMINDER_TIMEZONE = ZoneInfo("America/Sao_Paulo")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "ao", "as", "com", "da", "das", "de", "do", "dos", "e", "em", "na", "no",
    "o", "os", "para", "por", "um", "uma",
})


@dataclass(frozen=True)
class Reminder:
    id: int
    user_id: str
    text: str
    due_at: float
    status: str
    created_at: float
    updated_at: float
    recurrence_kind: str | None = None
    recurrence_weekday: int | None = None


@dataclass(frozen=True)
class ReminderRecurrence:
    kind: str
    weekday: int | None = None


@dataclass(frozen=True)
class ParsedReminderRequest:
    text: str
    due_at: float
    recurrence: ReminderRecurrence | None = None


@dataclass(frozen=True)
class ParsedReminderRescheduleRequest:
    text: str
    due_at: float


@dataclass(frozen=True)
class ParsedReminderDelayRequest:
    text: str
    delay_seconds: float


class ReminderServiceProtocol(Protocol):
    def create(
        self, user_id: str, text: str, due_at: float, *, recurrence: ReminderRecurrence | None = None,
        now: float | None = None,
    ) -> Reminder: ...

    def list_pending(self, user_id: str) -> tuple[Reminder, ...]: ...

    def find_pending_matches(self, user_id: str, text: str) -> tuple[Reminder, ...]: ...

    def find_delivered_for_snooze(self, user_id: str) -> tuple[Reminder, ...]: ...

    def cancel(self, user_id: str, reminder_id: int, *, now: float | None = None) -> bool: ...

    def get_due(self, now: float) -> tuple[Reminder, ...]: ...

    def mark_delivered(self, reminder_id: int, *, now: float | None = None) -> bool: ...

    def mark_acknowledged(self, reminder_id: int, *, now: float | None = None) -> bool: ...

    def reschedule(
        self, user_id: str, reminder_id: int, due_at: float, *, now: float | None = None,
    ) -> Reminder | None: ...

    def snooze(
        self, user_id: str, reminder_id: int, delay_seconds: float, *, now: float | None = None,
    ) -> Reminder | None: ...


class ReminderService:
    """Persiste lembretes locais por usuario com uma conexao SQLite compartilhada."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    def create(
        self, user_id: str, text: str, due_at: float, *, recurrence: ReminderRecurrence | None = None,
        now: float | None = None,
    ) -> Reminder:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Texto do lembrete nao pode ser vazio")
        recurrence_kind, recurrence_weekday = _recurrence_values(recurrence)
        timestamp = time.time() if now is None else now
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO reminders (
                        user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                    ) VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
                    """,
                    (user_id, normalized_text, due_at, timestamp, timestamp, recurrence_kind, recurrence_weekday),
                )
                reminder_id = cursor.lastrowid
        if reminder_id is None:
            raise RuntimeError("Nao foi possivel salvar o lembrete")
        return Reminder(
            reminder_id, user_id, normalized_text, due_at, "pending", timestamp, timestamp,
            recurrence_kind, recurrence_weekday,
        )

    def list_pending(self, user_id: str) -> tuple[Reminder, ...]:
        with self._lock:
            rows = self._get_connection().execute(
                """
                SELECT id, user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                FROM reminders
                WHERE user_id = ? AND status = 'pending'
                ORDER BY due_at ASC, id ASC
                """,
                (user_id,),
            ).fetchall()
        return tuple(self._to_reminder(row) for row in rows)

    def find_pending_matches(self, user_id: str, text: str) -> tuple[Reminder, ...]:
        query_terms = _terms(text)
        if not query_terms:
            return ()
        return tuple(
            reminder for reminder in self.list_pending(user_id)
            if query_terms.issubset(_terms(reminder.text))
        )

    def find_delivered_for_snooze(self, user_id: str) -> tuple[Reminder, ...]:
        """Retorna apenas entregas ainda sem confirmacao posterior do usuario."""
        with self._lock:
            rows = self._get_connection().execute(
                """
                SELECT id, user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                FROM reminders
                WHERE user_id = ? AND status = 'delivered'
                ORDER BY updated_at DESC, id DESC
                """,
                (user_id,),
            ).fetchall()
        return tuple(self._to_reminder(row) for row in rows)

    def cancel(self, user_id: str, reminder_id: int, *, now: float | None = None) -> bool:
        return self._update_status(reminder_id, "cancelled", now=now, user_id=user_id, from_status="pending")

    def get_due(self, now: float) -> tuple[Reminder, ...]:
        with self._lock:
            rows = self._get_connection().execute(
                """
                SELECT id, user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                FROM reminders
                WHERE status = 'pending' AND due_at <= ?
                ORDER BY due_at ASC, id ASC
                """,
                (now,),
            ).fetchall()
        return tuple(self._to_reminder(row) for row in rows)

    def mark_delivered(self, reminder_id: int, *, now: float | None = None) -> bool:
        """Confirma a entrega e rearma a proxima ocorrencia quando for recorrente."""
        timestamp = time.time() if now is None else now
        with self._lock:
            connection = self._get_connection()
            with connection:
                row = connection.execute(
                    """
                    SELECT id, user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                    FROM reminders WHERE id = ? AND status = 'pending'
                    """,
                    (reminder_id,),
                ).fetchone()
                if row is None:
                    return False
                reminder = self._to_reminder(row)
                if reminder.recurrence_kind is None:
                    cursor = connection.execute(
                        "UPDATE reminders SET status = 'delivered', updated_at = ? WHERE id = ? AND status = 'pending'",
                        (timestamp, reminder_id),
                    )
                else:
                    next_due_at = _next_recurrence_due(reminder, timestamp)
                    cursor = connection.execute(
                        """
                        UPDATE reminders SET due_at = ?, status = 'pending', updated_at = ?
                        WHERE id = ? AND status = 'pending'
                        """,
                        (next_due_at, timestamp, reminder_id),
                    )
        return cursor.rowcount == 1

    def mark_acknowledged(self, reminder_id: int, *, now: float | None = None) -> bool:
        return self._update_status(reminder_id, "acknowledged", now=now, from_status="delivered")

    def snooze(
        self, user_id: str, reminder_id: int, delay_seconds: float, *, now: float | None = None,
    ) -> Reminder | None:
        """Reagenda explicitamente um lembrete pendente sem criar recorrencia."""
        if delay_seconds <= 0:
            raise ValueError("delay_seconds deve ser maior que zero")
        timestamp = time.time() if now is None else now
        due_at = timestamp + delay_seconds
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE reminders
                    SET due_at = ?, status = 'pending', updated_at = ?
                    WHERE id = ? AND user_id = ? AND status IN ('pending', 'delivered')
                    """,
                    (due_at, timestamp, reminder_id, user_id),
                )
                if cursor.rowcount != 1:
                    return None
                row = connection.execute(
                    """
                    SELECT id, user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                    FROM reminders WHERE id = ? AND user_id = ?
                    """,
                    (reminder_id, user_id),
                ).fetchone()
        return self._to_reminder(row) if row is not None else None

    def reschedule(
        self, user_id: str, reminder_id: int, due_at: float, *, now: float | None = None,
    ) -> Reminder | None:
        """Edita somente um lembrete pendente pertencente ao usuario informado."""
        timestamp = time.time() if now is None else now
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE reminders SET due_at = ?, updated_at = ?
                    WHERE id = ? AND user_id = ? AND status = 'pending'
                    """,
                    (due_at, timestamp, reminder_id, user_id),
                )
                if cursor.rowcount != 1:
                    return None
                row = connection.execute(
                    """
                    SELECT id, user_id, text, due_at, status, created_at, updated_at, recurrence_kind, recurrence_weekday
                    FROM reminders WHERE id = ? AND user_id = ?
                    """,
                    (reminder_id, user_id),
                ).fetchone()
        return self._to_reminder(row) if row is not None else None

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _update_status(
        self,
        reminder_id: int,
        status: str,
        *,
        now: float | None,
        from_status: str,
        user_id: str | None = None,
    ) -> bool:
        if status not in REMINDER_STATUSES:
            raise ValueError("Status de lembrete invalido")
        timestamp = time.time() if now is None else now
        clauses = ["id = ?", "status = ?"]
        parameters: list[object] = [reminder_id, from_status]
        if user_id is not None:
            clauses.append("user_id = ?")
            parameters.append(user_id)
        parameters = [status, timestamp, *parameters]
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    f"UPDATE reminders SET status = ?, updated_at = ? WHERE {' AND '.join(clauses)}",
                    parameters,
                )
        return cursor.rowcount == 1

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._database_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    due_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    recurrence_kind TEXT,
                    recurrence_weekday INTEGER
                )
                """
            )
            columns = {row["name"] for row in self._connection.execute("PRAGMA table_info(reminders)")}
            if "recurrence_kind" not in columns:
                self._connection.execute("ALTER TABLE reminders ADD COLUMN recurrence_kind TEXT")
            if "recurrence_weekday" not in columns:
                self._connection.execute("ALTER TABLE reminders ADD COLUMN recurrence_weekday INTEGER")
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_user_status_due ON reminders (user_id, status, due_at, id)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_reminders_status_due ON reminders (status, due_at, id)"
            )
        return self._connection

    @staticmethod
    def _to_reminder(row: sqlite3.Row) -> Reminder:
        return Reminder(
            id=row["id"], user_id=row["user_id"], text=row["text"], due_at=row["due_at"],
            status=row["status"], created_at=row["created_at"], updated_at=row["updated_at"],
            recurrence_kind=row["recurrence_kind"], recurrence_weekday=row["recurrence_weekday"],
        )


def parse_reminder_request(text: str, *, now: datetime) -> ParsedReminderRequest | None:
    """Interpreta somente os formatos deterministas iniciais de lembrete."""
    normalized = _normalize(text)
    daily_match = DAILY_REMINDER_PATTERN.fullmatch(normalized)
    if daily_match is not None:
        due_at = _next_daily_due(now, daily_match.group("hour"), daily_match.group("minute"))
        reminder_text = daily_match.group("text").strip()
        if due_at is not None and reminder_text:
            return ParsedReminderRequest(
                text=reminder_text,
                due_at=due_at,
                recurrence=ReminderRecurrence(RECURRENCE_DAILY),
            )

    weekly_match = WEEKLY_REMINDER_PATTERN.fullmatch(normalized)
    if weekly_match is not None:
        weekday = WEEKDAY_BY_NAME.get(weekly_match.group("weekday"))
        due_at = _next_weekly_due(now, weekday, weekly_match.group("hour"), weekly_match.group("minute"))
        reminder_text = weekly_match.group("text").strip()
        if due_at is not None and reminder_text:
            return ParsedReminderRequest(
                text=reminder_text,
                due_at=due_at,
                recurrence=ReminderRecurrence(RECURRENCE_WEEKLY, weekday),
            )
    relative_match = RELATIVE_REMINDER_PATTERN.fullmatch(normalized)
    if relative_match is not None:
        amount = int(relative_match.group("amount"))
        reminder_text = relative_match.group("text").strip()
        if amount <= 0 or not reminder_text:
            return None
        unit = relative_match.group("unit")
        delta = timedelta(minutes=amount) if unit.startswith("minuto") else timedelta(hours=amount)
        return ParsedReminderRequest(text=reminder_text, due_at=(now + delta).timestamp())

    today_match = TODAY_REMINDER_PATTERN.fullmatch(normalized)
    if today_match is None:
        return None
    hour = int(today_match.group("hour"))
    minute = int(today_match.group("minute") or 0)
    reminder_text = today_match.group("text").strip()
    if hour > 23 or minute > 59 or not reminder_text:
        return None
    due_datetime = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due_datetime <= now:
        return None
    return ParsedReminderRequest(text=reminder_text, due_at=due_datetime.timestamp())


def parse_reminder_reschedule_request(text: str, *, now: datetime) -> ParsedReminderRescheduleRequest | None:
    match = REMINDER_RESCHEDULE_PATTERN.fullmatch(_normalize(text))
    if match is None:
        return None
    due_at = _next_daily_due(now, match.group("hour"), match.group("minute"), require_today=True)
    reminder_text = match.group("text").strip()
    if due_at is None or not reminder_text:
        return None
    return ParsedReminderRescheduleRequest(text=reminder_text, due_at=due_at)


def parse_reminder_delay_request(text: str) -> ParsedReminderDelayRequest | None:
    match = REMINDER_DELAY_PATTERN.fullmatch(_normalize(text))
    if match is None:
        return None
    amount = int(match.group("amount"))
    reminder_text = match.group("text").strip()
    if amount <= 0 or not reminder_text:
        return None
    return ParsedReminderDelayRequest(
        text=reminder_text,
        delay_seconds=_delay_seconds(amount, match.group("unit")),
    )


def parse_reminder_snooze_request(text: str) -> float | None:
    match = REMINDER_SNOOZE_PATTERN.fullmatch(_normalize(text))
    if match is None:
        return None
    amount = int(match.group("amount"))
    if amount <= 0:
        return None
    return _delay_seconds(amount, match.group("unit"))


def _recurrence_values(recurrence: ReminderRecurrence | None) -> tuple[str | None, int | None]:
    if recurrence is None:
        return None, None
    if recurrence.kind not in REMINDER_RECURRENCES:
        raise ValueError("Recorrencia de lembrete invalida")
    if recurrence.kind == RECURRENCE_WEEKLY:
        if recurrence.weekday is None or not 0 <= recurrence.weekday <= 6:
            raise ValueError("Dia da semana da recorrencia invalido")
        return recurrence.kind, recurrence.weekday
    if recurrence.weekday is not None:
        raise ValueError("Recorrencia diaria nao possui dia da semana")
    return recurrence.kind, None


def _next_recurrence_due(reminder: Reminder, delivered_at: float) -> float:
    scheduled = datetime.fromtimestamp(reminder.due_at, REMINDER_TIMEZONE)
    if reminder.recurrence_kind == RECURRENCE_DAILY:
        next_due = scheduled + timedelta(days=1)
        while next_due.timestamp() <= delivered_at:
            next_due += timedelta(days=1)
        return next_due.timestamp()
    if reminder.recurrence_kind == RECURRENCE_WEEKLY:
        next_due = scheduled + timedelta(days=7)
        while next_due.timestamp() <= delivered_at:
            next_due += timedelta(days=7)
        return next_due.timestamp()
    raise ValueError("Recorrencia de lembrete invalida")


def _next_daily_due(
    now: datetime, hour_text: str, minute_text: str | None, *, require_today: bool = False,
) -> float | None:
    hour = int(hour_text)
    minute = int(minute_text or 0)
    if hour > 23 or minute > 59:
        return None
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= now:
        if require_today:
            return None
        due += timedelta(days=1)
    return due.timestamp()


def _next_weekly_due(
    now: datetime, weekday: int | None, hour_text: str, minute_text: str | None,
) -> float | None:
    if weekday is None:
        return None
    hour = int(hour_text)
    minute = int(minute_text or 0)
    if hour > 23 or minute > 59:
        return None
    due = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    due += timedelta(days=(weekday - now.weekday()) % 7)
    if due <= now:
        due += timedelta(days=7)
    return due.timestamp()


def _delay_seconds(amount: int, unit: str) -> float:
    delta = timedelta(minutes=amount) if unit.startswith("minuto") else timedelta(hours=amount)
    return delta.total_seconds()


def is_reminder_request(text: str) -> bool:
    return REMINDER_PREFIX_PATTERN.match(_normalize(text)) is not None


def is_reminder_list_request(text: str) -> bool:
    return REMINDER_LIST_PATTERN.fullmatch(_normalize(text)) is not None


def parse_reminder_cancel_request(text: str) -> str | None:
    match = REMINDER_CANCEL_PATTERN.fullmatch(_normalize(text))
    if match is None:
        return None
    reminder_text = match.group("text").strip()
    return reminder_text or None


def _terms(text: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_PATTERN.findall(_normalize(text)) if len(token) >= 2 and token not in _STOPWORDS)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text).casefold()
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^a-z0-9:]+", " ", without_accents).split())
