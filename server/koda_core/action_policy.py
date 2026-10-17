"""Politica deterministica e independente de dispositivo para acoes do Koda."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from .contracts import Action


class ActionPermission(StrEnum):
    """Nivel de permissao necessario antes de uma acao poder ser executada."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


_ALLOWED_ACTION_TYPES = frozenset({"set_volume", "show_text", "expression", "get_time", "search"})
_CONFIRM_ACTION_TYPES = frozenset({
    "send_message",
    "start_call",
    "modify_file",
    "run_task",
    "change_configuration",
})
_DENIED_ACTION_TYPES = frozenset({
    "purchase",
    "access_bank",
    "bank_transfer",
    "bypass_biometric",
})


class ActionPolicy:
    """Classifica actions sem depender de LLM, transporte ou dispositivo."""

    def evaluate(self, action: Mapping[str, object]) -> ActionPermission:
        action_type = action.get("type")
        if not isinstance(action_type, str) or not action_type:
            return ActionPermission.DENY
        if action_type in _DENIED_ACTION_TYPES:
            return ActionPermission.DENY
        if action_type in _ALLOWED_ACTION_TYPES:
            return ActionPermission.ALLOW
        if action_type in _CONFIRM_ACTION_TYPES:
            return ActionPermission.CONFIRM
        return ActionPermission.CONFIRM


@dataclass(frozen=True)
class ActionAuthorization:
    """Resultado de uma proposta; somente ALLOW pode seguir imediatamente."""

    permission: ActionPermission
    action: Action
    authorization_id: str | None = None

    @property
    def can_execute(self) -> bool:
        return self.permission is ActionPermission.ALLOW


class PendingActionAuthorizations:
    """Mantem propostas CONFIRM em RAM ate aprovacao ou recusa explicita."""

    def __init__(
        self,
        policy: ActionPolicy | None = None,
        *,
        identifier_factory: Callable[[], str] = lambda: uuid4().hex,
    ) -> None:
        self._policy = policy or ActionPolicy()
        self._identifier_factory = identifier_factory
        self._pending: dict[str, Action] = {}
        self._lock = RLock()

    def propose(self, action: Mapping[str, object]) -> ActionAuthorization:
        copied_action = dict(action)
        permission = self._policy.evaluate(copied_action)
        if permission is not ActionPermission.CONFIRM:
            return ActionAuthorization(permission=permission, action=copied_action)
        authorization_id = self._identifier_factory()
        with self._lock:
            self._pending[authorization_id] = copied_action
        return ActionAuthorization(
            permission=permission,
            action=copied_action,
            authorization_id=authorization_id,
        )

    def get_pending(self, authorization_id: str) -> Action | None:
        with self._lock:
            action = self._pending.get(authorization_id)
            return dict(action) if action is not None else None

    def approve(self, authorization_id: str) -> Action | None:
        """Libera a action apenas apos autorizacao explicita; nao a executa."""
        with self._lock:
            action = self._pending.pop(authorization_id, None)
            return dict(action) if action is not None else None

    def reject(self, authorization_id: str) -> bool:
        """Remove uma proposta pendente sem devolve-la para execucao."""
        with self._lock:
            return self._pending.pop(authorization_id, None) is not None
