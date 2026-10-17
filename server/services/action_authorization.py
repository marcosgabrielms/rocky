"""Sessao local de autorizacao explicita para acoes futuras do Koda."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import time
import unicodedata
from threading import RLock

from koda_core import ActionPermission, ActionPolicy, PendingActionAuthorizations
from koda_core.contracts import Action


AUTHORIZATION_TTL_SECONDS = 120.0
_APPROVAL_PHRASES = frozenset({"sim", "pode", "pode fazer", "autorizo", "confirma"})
_REJECTION_PHRASES = frozenset({"nao", "cancela", "nao faca"})


class AuthorizationStatus(StrEnum):
    ALLOWED = "allowed"
    PENDING = "pending"
    DENIED = "denied"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class AuthorizationProposal:
    status: AuthorizationStatus
    permission: ActionPermission
    action: Action
    authorization_id: str | None = None
    expires_at: float | None = None


@dataclass(frozen=True)
class AuthorizationResolution:
    status: AuthorizationStatus
    action: Action | None = None


@dataclass(frozen=True)
class _PendingAuthorization:
    authorization_id: str
    action: Action
    expires_at: float


class ActionAuthorizationSession:
    """Mantem no maximo uma autorizacao pendente, temporaria, por usuario."""

    def __init__(
        self,
        policy: ActionPolicy | None = None,
        pending_authorizations: PendingActionAuthorizations | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_seconds: float = AUTHORIZATION_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("authorization_ttl_seconds deve ser maior que zero")
        self._policy = policy or ActionPolicy()
        self._pending_authorizations = pending_authorizations or PendingActionAuthorizations(self._policy)
        self._clock = clock
        self._ttl_seconds = ttl_seconds
        self._pending_by_user: dict[str, _PendingAuthorization] = {}
        self._lock = RLock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def propose(
        self, user_id: str, action: Mapping[str, object], *, now: float | None = None,
    ) -> AuthorizationProposal:
        with self._lock:
            timestamp = self._clock() if now is None else now
            self._expire(user_id, timestamp)
            copied_action = dict(action)
            permission = self._policy.evaluate(copied_action)
            if permission is ActionPermission.ALLOW:
                return AuthorizationProposal(AuthorizationStatus.ALLOWED, permission, copied_action)
            if permission is ActionPermission.DENY:
                return AuthorizationProposal(AuthorizationStatus.DENIED, permission, copied_action)

            existing = self._pending_by_user.get(user_id)
            if existing is not None:
                return AuthorizationProposal(
                    AuthorizationStatus.PENDING,
                    permission,
                    dict(existing.action),
                    existing.authorization_id,
                    existing.expires_at,
                )

            authorization = self._pending_authorizations.propose(copied_action)
            if authorization.authorization_id is None:
                raise RuntimeError("Acao que exige confirmacao nao recebeu identificador")
            expires_at = timestamp + self._ttl_seconds
            self._pending_by_user[user_id] = _PendingAuthorization(
                authorization.authorization_id,
                copied_action,
                expires_at,
            )
            return AuthorizationProposal(
                AuthorizationStatus.PENDING,
                permission,
                copied_action,
                authorization.authorization_id,
                expires_at,
            )

    def resolve(
        self, user_id: str, text: str, *, now: float | None = None,
    ) -> AuthorizationResolution | None:
        with self._lock:
            timestamp = self._clock() if now is None else now
            phrase = _normalize_phrase(text)
            if phrase not in _APPROVAL_PHRASES and phrase not in _REJECTION_PHRASES:
                self._expire(user_id, timestamp)
                return None

            pending = self._pending_by_user.get(user_id)
            if pending is None:
                return None
            if pending.expires_at <= timestamp:
                self._pending_authorizations.reject(pending.authorization_id)
                self._pending_by_user.pop(user_id, None)
                return AuthorizationResolution(AuthorizationStatus.EXPIRED)

            self._pending_by_user.pop(user_id, None)
            if phrase in _APPROVAL_PHRASES:
                action = self._pending_authorizations.approve(pending.authorization_id)
                if action is None:
                    return AuthorizationResolution(AuthorizationStatus.EXPIRED)
                return AuthorizationResolution(AuthorizationStatus.APPROVED, action)
            self._pending_authorizations.reject(pending.authorization_id)
            return AuthorizationResolution(AuthorizationStatus.REJECTED)

    def get_pending(self, user_id: str, *, now: float | None = None) -> AuthorizationProposal | None:
        with self._lock:
            timestamp = self._clock() if now is None else now
            if self._expire(user_id, timestamp):
                return None
            pending = self._pending_by_user.get(user_id)
            if pending is None:
                return None
            return AuthorizationProposal(
                AuthorizationStatus.PENDING,
                ActionPermission.CONFIRM,
                dict(pending.action),
                pending.authorization_id,
                pending.expires_at,
            )

    def _expire(self, user_id: str, now: float) -> bool:
        pending = self._pending_by_user.get(user_id)
        if pending is None or pending.expires_at > now:
            return False
        self._pending_authorizations.reject(pending.authorization_id)
        self._pending_by_user.pop(user_id, None)
        return True


def _normalize_phrase(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text).casefold()
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join("".join(character if character.isalnum() else " " for character in without_accents).split())
