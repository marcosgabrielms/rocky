"""Armazenamento em memÃ³ria do contexto curto de conversa."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Protocol

from services.llm.base import ConversationMessage


MAX_CONTEXT_MESSAGES = 6
CONTEXT_TTL_SECONDS = 5 * 60


@dataclass(frozen=True)
class RecentContextPolicy:
    """Define a retencao do contexto recente, nunca memoria de longo prazo."""

    max_messages: int = MAX_CONTEXT_MESSAGES
    ttl_seconds: float = CONTEXT_TTL_SECONDS

    def __post_init__(self) -> None:
        if self.max_messages <= 0:
            raise ValueError("max_messages deve ser maior que zero")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds deve ser maior que zero")


DEFAULT_RECENT_CONTEXT_POLICY = RecentContextPolicy()


@dataclass
class ConversationHistory:
    messages: list[ConversationMessage]
    last_interaction_at: float


class ContextStore(Protocol):
    """Contrato minimo para o historico curto de uma conversa."""

    def get_context(self, user_id: str, now: float) -> tuple[ConversationMessage, ...]: ...

    def add_exchange(self, user_id: str, user_text: str, assistant_text: str, now: float) -> None: ...

    def clear(self, user_id: str) -> None: ...

    def purge_expired(self, now: float) -> int: ...


class ConversationContextStore:
    """MantÃ©m contexto curto isolado por identidade lÃ³gica, somente em RAM."""

    def __init__(
        self,
        *,
        policy: RecentContextPolicy = DEFAULT_RECENT_CONTEXT_POLICY,
        max_messages: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._histories: dict[str, ConversationHistory] = {}
        self._policy = _resolve_policy(policy, max_messages, ttl_seconds)

    @property
    def policy(self) -> RecentContextPolicy:
        return self._policy

    def get_context(self, user_id: str, now: float) -> tuple[ConversationMessage, ...]:
        self.purge_expired(now)
        history = self._histories.get(user_id)
        if history is None:
            return ()
        return tuple(history.messages)

    def add_exchange(self, user_id: str, user_text: str, assistant_text: str, now: float) -> None:
        self.purge_expired(now)
        history = self._histories.get(user_id)
        messages = list(history.messages) if history is not None else []
        messages.extend(
            (
                ConversationMessage(role="user", content=user_text),
                ConversationMessage(role="assistant", content=assistant_text),
            )
        )
        self._histories[user_id] = ConversationHistory(
            messages=messages[-self._policy.max_messages :],
            last_interaction_at=now,
        )

    def clear(self, user_id: str) -> None:
        self._histories.pop(user_id, None)

    def purge_expired(self, now: float) -> int:
        expired_user_ids = [
            user_id
            for user_id, history in self._histories.items()
            if now - history.last_interaction_at >= self._policy.ttl_seconds
        ]
        for user_id in expired_user_ids:
            self.clear(user_id)
        return len(expired_user_ids)


class SqliteConversationContextStore:
    """Mantem o contexto curto em SQLite local, isolado por identidade logica."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        policy: RecentContextPolicy = DEFAULT_RECENT_CONTEXT_POLICY,
        max_messages: int | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        self._database_path = Path(database_path)
        self._policy = _resolve_policy(policy, max_messages, ttl_seconds)
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    @property
    def policy(self) -> RecentContextPolicy:
        return self._policy

    def get_context(self, user_id: str, now: float) -> tuple[ConversationMessage, ...]:
        with self._lock:
            connection = self._get_connection()
            self.purge_expired(now)

            rows = connection.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE user_id = ?
                ORDER BY sequence ASC
                """,
                (user_id,),
            ).fetchall()

        return tuple(ConversationMessage(role=row["role"], content=row["content"]) for row in rows)

    def add_exchange(self, user_id: str, user_text: str, assistant_text: str, now: float) -> None:
        with self._lock:
            connection = self._get_connection()
            self.purge_expired(now)
            with connection:
                connection.executemany(
                    """
                    INSERT INTO conversation_messages (user_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (user_id, "user", user_text, now),
                        (user_id, "assistant", assistant_text, now),
                    ),
                )
                connection.execute(
                    """
                    DELETE FROM conversation_messages
                    WHERE user_id = ?
                      AND sequence NOT IN (
                          SELECT sequence
                          FROM conversation_messages
                          WHERE user_id = ?
                          ORDER BY sequence DESC
                          LIMIT ?
                      )
                    """,
                    (user_id, user_id, self._policy.max_messages),
                )

    def clear(self, user_id: str) -> None:
        with self._lock:
            connection = self._get_connection()
            with connection:
                connection.execute("DELETE FROM conversation_messages WHERE user_id = ?", (user_id,))

    def purge_expired(self, now: float) -> int:
        """Remove historicos inativos sem depender do retorno de cada usuario."""
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    """
                    DELETE FROM conversation_messages
                    WHERE user_id IN (
                        SELECT user_id
                        FROM conversation_messages
                        GROUP BY user_id
                        HAVING ? - MAX(timestamp) >= ?
                    )
                    """,
                    (now, self._policy.ttl_seconds),
                )
        return cursor.rowcount

    def close(self) -> None:
        """Fecha a conexao local quando o processo de teste termina."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _get_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._database_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self._database_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    user_id TEXT NOT NULL,
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_conversation_messages_user_sequence
                ON conversation_messages (user_id, sequence)
                """
            )
        return self._connection


def _resolve_policy(
    policy: RecentContextPolicy,
    max_messages: int | None,
    ttl_seconds: float | None,
) -> RecentContextPolicy:
    if max_messages is None and ttl_seconds is None:
        return policy
    return RecentContextPolicy(
        max_messages=policy.max_messages if max_messages is None else max_messages,
        ttl_seconds=policy.ttl_seconds if ttl_seconds is None else ttl_seconds,
    )
