"""Composição única dos componentes conversacionais compartilhados do Koda."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from koda_core import ActionPolicy, ConversationCoreAdapter, PendingActionAuthorizations
from services.action_authorization import (
    AUTHORIZATION_TTL_SECONDS,
    ActionAuthorizationSession,
)
from services.command_session import CommandSession
from services.conversation import ConversationManager
from services.conversation_context import (
    DEFAULT_RECENT_CONTEXT_POLICY,
    ContextStore,
    RecentContextPolicy,
    SqliteConversationContextStore,
)
from services.interaction import KodaInteractionService
from services.long_term_memory import LongTermMemoryStore, LongTermMemoryStoreProtocol
from services.reminders import ReminderService, ReminderServiceProtocol
from services.reminder_scheduler import ReminderDelivery, ReminderScheduler


CONTEXT_DATABASE_PATH = Path(__file__).parent / "local" / "context" / "koda_context.sqlite3"
LONG_TERM_MEMORY_DATABASE_PATH = Path(__file__).parent / "local" / "memory" / "koda_memory.sqlite3"
REMINDER_DATABASE_PATH = Path(__file__).parent / "local" / "reminders" / "koda_reminders.sqlite3"


@dataclass(frozen=True)
class KodaRuntime:
    """Agrupa o estado compartilhado necessário à fronteira HTTP/dispositivo."""

    conversation_manager: ConversationManager
    conversation_core_adapter: ConversationCoreAdapter
    command_session: CommandSession
    interaction_service: KodaInteractionService
    conversation_context_store: ContextStore
    long_term_memory_store: LongTermMemoryStoreProtocol
    reminder_service: ReminderServiceProtocol
    reminder_scheduler: ReminderScheduler
    action_policy: ActionPolicy
    pending_action_authorizations: PendingActionAuthorizations
    action_authorization_session: ActionAuthorizationSession

    @classmethod
    def create(
        cls,
        llm_client: object,
        *,
        context_database_path: str | Path = CONTEXT_DATABASE_PATH,
        context_policy: RecentContextPolicy = DEFAULT_RECENT_CONTEXT_POLICY,
        memory_database_path: str | Path = LONG_TERM_MEMORY_DATABASE_PATH,
        reminder_database_path: str | Path = REMINDER_DATABASE_PATH,
        reminder_delivery: ReminderDelivery | None = None,
        reminder_scheduler_clock: Callable[[], float] = time.time,
        reminder_scheduler_interval_seconds: float = 30.0,
        authorization_clock: Callable[[], float] = time.monotonic,
        authorization_ttl_seconds: float = AUTHORIZATION_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        memory_clock: Callable[[], float] = time.time,
    ) -> "KodaRuntime":
        conversation_context_store = SqliteConversationContextStore(
            context_database_path,
            policy=context_policy,
        )
        conversation_context_store.purge_expired(clock())
        long_term_memory_store = LongTermMemoryStore(memory_database_path)
        reminder_service = ReminderService(reminder_database_path)
        reminder_scheduler = ReminderScheduler(
            reminder_service,
            reminder_delivery,
            clock=reminder_scheduler_clock,
            interval_seconds=reminder_scheduler_interval_seconds,
        )
        action_policy = ActionPolicy()
        pending_action_authorizations = PendingActionAuthorizations(action_policy)
        action_authorization_session = ActionAuthorizationSession(
            action_policy,
            pending_action_authorizations,
            clock=authorization_clock,
            ttl_seconds=authorization_ttl_seconds,
        )
        conversation_manager = ConversationManager(
            0,
            llm_client,
            clock=clock,
            context_store=conversation_context_store,
            memory_store=long_term_memory_store,
            memory_clock=memory_clock,
            reminder_service=reminder_service,
            action_authorization_session=action_authorization_session,
        )
        conversation_core_adapter = ConversationCoreAdapter(conversation_manager)
        return cls(
            conversation_manager=conversation_manager,
            conversation_core_adapter=conversation_core_adapter,
            command_session=CommandSession(),
            interaction_service=KodaInteractionService(conversation_core_adapter),
            conversation_context_store=conversation_context_store,
            long_term_memory_store=long_term_memory_store,
            reminder_service=reminder_service,
            reminder_scheduler=reminder_scheduler,
            action_policy=action_policy,
            pending_action_authorizations=pending_action_authorizations,
            action_authorization_session=action_authorization_session,
        )
