"""Memoria local persistente, salva somente por pedido explicito do usuario."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from threading import RLock
import time
from typing import Protocol
import unicodedata


MEMORY_CATEGORIES = frozenset({"preference", "project", "routine", "decision", "place", "taste", "note"})
MEMORY_REQUEST_PATTERN = re.compile(
    r"^(?:lembra(?:[-\s]se)?|lembre(?:[-\s]se)?|guarda|guarde)\s+(?:que\s+)?(?P<content>.+)$",
    re.IGNORECASE,
)
REMINDER_REQUEST_PATTERN = re.compile(r"^(?:me\s+)?lembra(?:[-\s]me)?\s+de\b", re.IGNORECASE)
MEMORY_QUERY_PATTERN = re.compile(r"^o que voce lembra (?:sobre|de) mim$", re.IGNORECASE)
MEMORY_FORGET_PATTERN = re.compile(r"^esquece\s+que\s+(?P<content>.+)$", re.IGNORECASE)
SENSITIVE_MEMORY_PATTERN = re.compile(
    r"\b(?:senha|password|token|api[_\s-]?key|chave\s+(?:de\s+)?api|"
    r"cart[aã]o|conta\s+banc[aá]ria|biometria|impress[aã]o\s+digital|"
    r"documento\s+completo|n[uú]mero\s+do\s+cpf|cpf)\b",
    re.IGNORECASE,
)
MAX_MEMORY_SEARCH_RESULTS = 3
_LEXICAL_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_LEXICAL_STOPWORDS = frozenset({
    "a", "ao", "as", "com", "da", "das", "de", "do", "dos", "e", "em", "eu", "me",
    "minha", "meu", "na", "nas", "no", "nos", "o", "os", "para", "por", "que", "sobre",
    "um", "uma", "voce",
})
_CATEGORY_TERMS = {
    "preference": frozenset({"preference", "preferencia", "preferencias"}),
    "project": frozenset({"project", "projeto", "projetos"}),
    "routine": frozenset({"routine", "rotina", "rotinas"}),
    "decision": frozenset({"decision", "decisao", "decisoes"}),
    "place": frozenset({"place", "lugar", "lugares"}),
    "taste": frozenset({"taste", "gosto", "gostos"}),
    "note": frozenset({"note", "nota", "notas"}),
}


@dataclass(frozen=True)
class LongTermMemory:
    id: int
    user_id: str
    category: str
    content: str
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class ExplicitMemoryRequest:
    content: str
    category: str


class LongTermMemoryStoreProtocol(Protocol):
    def add(self, user_id: str, category: str, content: str, *, now: float | None = None) -> LongTermMemory: ...

    def list(self, user_id: str) -> tuple[LongTermMemory, ...]: ...

    def search(self, user_id: str, query: str, *, limit: int = MAX_MEMORY_SEARCH_RESULTS) -> tuple[LongTermMemory, ...]: ...

    def delete(self, user_id: str, memory_id: int) -> bool: ...

    def clear(self, user_id: str) -> int: ...


class LongTermMemoryStore:
    """Armazena memoria explicita em SQLite, isolada por user_id."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = RLock()

    def add(self, user_id: str, category: str, content: str, *, now: float | None = None) -> LongTermMemory:
        normalized_category = category.strip().casefold()
        normalized_content = content.strip()
        if normalized_category not in MEMORY_CATEGORIES:
            raise ValueError("Categoria de memoria invalida")
        if not normalized_content:
            raise ValueError("Conteudo da memoria nao pode ser vazio")

        timestamp = time.time() if now is None else now
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO long_term_memories (user_id, category, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, normalized_category, normalized_content, timestamp, timestamp),
                )
                memory_id = cursor.lastrowid
        if memory_id is None:
            raise RuntimeError("Nao foi possivel salvar a memoria")
        return LongTermMemory(memory_id, user_id, normalized_category, normalized_content, timestamp, timestamp)

    def list(self, user_id: str) -> tuple[LongTermMemory, ...]:
        with self._lock:
            rows = self._get_connection().execute(
                """
                SELECT id, user_id, category, content, created_at, updated_at
                FROM long_term_memories
                WHERE user_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (user_id,),
            ).fetchall()
        return tuple(self._to_memory(row) for row in rows)

    def search(
        self, user_id: str, query: str, *, limit: int = MAX_MEMORY_SEARCH_RESULTS
    ) -> tuple[LongTermMemory, ...]:
        """Recupera poucas memorias por sobreposicao lexical, sem embeddings."""
        if limit <= 0:
            return ()
        query_terms = _lexical_terms(query)
        ranked: list[tuple[int, float, int, LongTermMemory]] = []
        for memory in self.list(user_id):
            memory_terms = _lexical_terms(memory.content).union(_CATEGORY_TERMS[memory.category])
            matches = len(query_terms.intersection(memory_terms))
            # Preferencias gerais podem ser uteis em qualquer resposta, mesmo sem termo em comum.
            score = matches * 100 + (1 if memory.category == "preference" else 0)
            if score:
                ranked.append((score, memory.updated_at, memory.id, memory))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        maximum = min(limit, MAX_MEMORY_SEARCH_RESULTS)
        return tuple(item[3] for item in ranked[:maximum])

    def delete(self, user_id: str, memory_id: int) -> bool:
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute(
                    "DELETE FROM long_term_memories WHERE id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
        return cursor.rowcount == 1

    def clear(self, user_id: str) -> int:
        with self._lock:
            connection = self._get_connection()
            with connection:
                cursor = connection.execute("DELETE FROM long_term_memories WHERE user_id = ?", (user_id,))
        return cursor.rowcount

    def close(self) -> None:
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
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_long_term_memories_user_created
                ON long_term_memories (user_id, created_at, id)
                """
            )
        return self._connection

    @staticmethod
    def _to_memory(row: sqlite3.Row) -> LongTermMemory:
        return LongTermMemory(
            id=row["id"],
            user_id=row["user_id"],
            category=row["category"],
            content=row["content"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def parse_explicit_memory_request(text: str) -> ExplicitMemoryRequest | None:
    """Extrai apenas pedidos declarados de salvar uma memoria."""
    match = MEMORY_REQUEST_PATTERN.fullmatch(text.strip())
    if match is None:
        return None
    content = match.group("content").strip()
    if not content:
        return None
    return ExplicitMemoryRequest(content=content, category=infer_memory_category(content))


def is_memory_query(text: str) -> bool:
    return MEMORY_QUERY_PATTERN.fullmatch(_normalize_memory_text(text)) is not None


def parse_memory_forget_request(text: str) -> str | None:
    match = MEMORY_FORGET_PATTERN.fullmatch(_normalize_memory_text(text))
    if match is None:
        return None
    content = match.group("content").strip()
    return content or None


def is_reminder_request(text: str) -> bool:
    return REMINDER_REQUEST_PATTERN.match(text.strip()) is not None


def contains_sensitive_memory_content(content: str) -> bool:
    return SENSITIVE_MEMORY_PATTERN.search(content) is not None


def infer_memory_category(content: str) -> str:
    normalized = content.casefold()
    if "projeto" in normalized:
        return "project"
    if "rotina" in normalized:
        return "routine"
    if "decid" in normalized:
        return "decision"
    if "moro" in normalized or "lugar" in normalized:
        return "place"
    if "gosto" in normalized:
        return "taste"
    if "prefiro" in normalized or "prefer" in normalized:
        return "preference"
    return "note"


def normalized_memory_content(content: str) -> str:
    return _normalize_memory_text(content)


def _lexical_terms(text: str) -> frozenset[str]:
    return frozenset(
        token for token in _LEXICAL_TOKEN_PATTERN.findall(_normalize_memory_text(text))
        if len(token) >= 3 and token not in _LEXICAL_STOPWORDS
    )


def _normalize_memory_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text).casefold()
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())
