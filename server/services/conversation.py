"""Estado conversacional em memória, separado por dispositivo Rocky."""

import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from models.actions import BackendResponse, expression, idle_response, set_device_state, set_volume, show_text
from services.commands import (
    DeviceStateIntent,
    detect_command_intent,
    detect_device_state_intent,
    extract_volume_percent,
    get_current_time,
    normalize_text,
)
from services.conversation_context import (
    CONTEXT_TTL_SECONDS,
    MAX_CONTEXT_MESSAGES,
    ContextStore,
    ConversationContextStore,
)
from services.long_term_memory import (
    LongTermMemoryStoreProtocol,
    contains_sensitive_memory_content,
    is_memory_query,
    normalized_memory_content,
    parse_explicit_memory_request,
    parse_memory_forget_request,
)
from services.llm import ConversationMessage, LLMClient, LLMError, SentenceBuffer, StreamingLLMClient
from services.reminders import (
    ReminderServiceProtocol,
    is_reminder_list_request,
    is_reminder_request,
    parse_reminder_cancel_request,
    parse_reminder_delay_request,
    parse_reminder_request,
    parse_reminder_reschedule_request,
    parse_reminder_snooze_request,
)
from services.latency import latency_metric, mark_latency
from services.runtime_log import error as log_error, info, technical
from services.action_authorization import ActionAuthorizationSession, AuthorizationStatus


WAKE_PREFIX = re.compile(r"^\W*(?P<wake>\w+)(?=$|\W)(?P<command>.*)$", re.DOTALL)
LLM_UNAVAILABLE_MESSAGE = "Não consegui acessar meu modelo de linguagem agora."
LLM_INTERRUPTED_MESSAGE = "Não consegui concluir minha resposta."
COMMAND_FILLERS = frozenset({"aham", "uhum", "hm"})
NON_LEXICAL_VOCALIZATION = re.compile(r"^(?P<sound>[akmh])(?P=sound){5,}$")
MAX_MEMORIES_IN_LIST_RESPONSE = 3
MEMORY_CONTEXT_HEADER = (
    "Memorias explicitas do usuario. Use somente quando forem relevantes para responder; "
    "nao as apresente como uma nova fala do usuario:"
)
REMINDER_TIMEZONE = ZoneInfo("America/Sao_Paulo")


def is_wake_phrase(text: str) -> bool:
    return extract_wake_command(text) == ""


def canonical_wake_word(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text).casefold()
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def extract_wake_command(text: str) -> str | None:
    """Somente Koda como primeiro token; None indica ausencia de wake valida."""
    match = WAKE_PREFIX.match(text)
    if match is None or canonical_wake_word(match.group("wake")) != "koda":
        return None
    return re.sub(r"^\W+", "", match.group("command")).strip()


class InteractionKind(StrEnum):
    EMPTY = "empty"
    WAKE = "wake"
    DETERMINISTIC = "deterministic"
    LLM = "llm"


class InteractionMode(StrEnum):
    WAKE = "wake"
    COMMAND = "command"


@dataclass(frozen=True)
class InteractionDecision:
    kind: InteractionKind
    text: str
    device_id: str
    direct_response: BackendResponse | None = None
    context: tuple[ConversationMessage, ...] = ()
    user_id: str = ""


@dataclass(frozen=True)
class StreamedConversationResponse:
    """Resposta final e sentenças prontas para uma futura saída em tempo real."""

    response: BackendResponse
    sentences: tuple[str, ...]
    stream_failed: bool = False


class ConversationManager:
    def __init__(
        self,
        command_window_ms: int,
        llm_client: LLMClient,
        clock: Callable[[], float] = time.monotonic,
        context_store: ContextStore | None = None,
        memory_store: LongTermMemoryStoreProtocol | None = None,
        memory_clock: Callable[[], float] = time.time,
        reminder_service: ReminderServiceProtocol | None = None,
        reminder_now: Callable[[], datetime] | None = None,
        action_authorization_session: ActionAuthorizationSession | None = None,
    ) -> None:
        # Argumento legado; somente o corpo fisico controla a espera por fala.
        del command_window_ms
        self._context_store = context_store or ConversationContextStore()
        self._memory_store = memory_store
        self._reminder_service = reminder_service
        self._action_authorization_session = action_authorization_session
        self._llm_client = llm_client
        self._clock = clock
        self._memory_clock = memory_clock
        self._reminder_now = reminder_now or (lambda: datetime.now(REMINDER_TIMEZONE))

    @property
    def context_store(self) -> ContextStore:
        return self._context_store

    @property
    def memory_store(self) -> LongTermMemoryStoreProtocol | None:
        return self._memory_store

    @property
    def reminder_service(self) -> ReminderServiceProtocol | None:
        return self._reminder_service

    @property
    def action_authorization_session(self) -> ActionAuthorizationSession | None:
        return self._action_authorization_session

    def propose_action(self, action: dict[str, object], *, user_id: str) -> BackendResponse:
        """Aplica a politica a uma acao futura, sem executa-la."""
        if self._action_authorization_session is None:
            return self._authorization_response("denied")

        proposal = self._action_authorization_session.propose(user_id, action)
        if proposal.status is AuthorizationStatus.ALLOWED:
            return self._authorization_response("allowed", action=proposal.action)
        if proposal.status is AuthorizationStatus.PENDING:
            return self._authorization_response("pending")
        return self._authorization_response("denied")

    def capture_request_time(self) -> float:
        """Timestamp funcional em segundos, independente do tracker de diagnóstico."""
        return self._clock()

    def build_response(
        self, text: str, device_id: str, *, user_id: str | None = None,
        received_at: float | None = None,
        interaction_mode: InteractionMode | None = None, low_confidence: bool = False,
    ) -> BackendResponse:
        return self.build_response_from_decision(
            self.classify(text, device_id, user_id=user_id, received_at=received_at, interaction_mode=interaction_mode,
                          low_confidence=low_confidence)
        )

    def build_response_from_decision(self, decision: InteractionDecision) -> BackendResponse:
        if decision.direct_response is not None:
            return decision.direct_response
        if decision.kind is not InteractionKind.LLM:
            raise ValueError("A non-LLM interaction decision requires a direct response")

        try:
            result = self._llm_client.ask(decision.text, context=decision.context)
        except LLMError as error:
            return self._unavailable_llm_response(error)

        if result.text.strip():
            self._context_store.add_exchange(decision.user_id, decision.text, result.text, self._clock())

        return self._idle_llm_response(result.text)

    def build_stream_response(
        self,
        text: str,
        device_id: str,
        *,
        user_id: str | None = None,
        on_sentence: Callable[[str, int, int], None] | None = None,
    ) -> StreamedConversationResponse:
        """Executa uma única geração em streaming sem mudar o contrato HTTP atual."""
        return self.build_stream_response_from_decision(
            self.classify(text, device_id, user_id=user_id), on_sentence
        )

    def build_stream_response_from_decision(
        self,
        decision: InteractionDecision,
        on_sentence: Callable[[str, int, int], None] | None = None,
    ) -> StreamedConversationResponse:
        if decision.direct_response is not None:
            return StreamedConversationResponse(response=decision.direct_response, sentences=())
        if decision.kind is not InteractionKind.LLM:
            raise ValueError("A non-LLM interaction decision requires a direct response")

        if not isinstance(self._llm_client, StreamingLLMClient):
            return StreamedConversationResponse(
                response=self._unavailable_llm_response(LLMError("Streaming LLM is not available")),
                sentences=(),
            )

        started_at = time.monotonic()
        first_token_at: float | None = None
        first_sentence_at: float | None = None
        buffer = SentenceBuffer(preserve_line_breaks=True)
        sentences: list[str] = []
        received_text = False
        technical("[RT] llm_request_start")
        mark_latency("llm_request_start")
        try:
            for chunk in self._llm_client.stream_response(decision.text, context=decision.context):
                if not chunk:
                    continue
                now = time.monotonic()
                if first_token_at is None:
                    first_token_at = now
                    mark_latency("llm_first_token")
                    latency_metric("llm_first_token_ms", "llm_request_start", "llm_first_token")
                    technical(f"[RT] llm_first_token_ms={round((now - started_at) * 1000)}")
                received_text = True
                new_sentences = buffer.push(chunk)
                if new_sentences and first_sentence_at is None:
                    first_sentence_at = now
                    technical(f"[RT] first_sentence_ms={round((now - started_at) * 1000)}")
                self._append_sentences(sentences, new_sentences, started_at, on_sentence)
        except LLMError as error:
            if not received_text:
                self._print_stream_metrics(started_at, len(sentences))
                return StreamedConversationResponse(
                    response=self._unavailable_llm_response(error),
                    sentences=(),
                    stream_failed=True,
                )

            residual = buffer.finish()
            if residual:
                self._append_sentences(sentences, (residual,), started_at, on_sentence)
            partial_text = " ".join(sentences).strip()
            self._print_stream_metrics(started_at, len(sentences))
            return StreamedConversationResponse(
                response=self._idle_llm_response(f"{partial_text} {LLM_INTERRUPTED_MESSAGE}"),
                sentences=tuple(sentences),
                stream_failed=True,
            )

        residual = buffer.finish()
        if residual:
            if first_sentence_at is None:
                technical(f"[RT] first_sentence_ms={round((time.monotonic() - started_at) * 1000)}")
            self._append_sentences(sentences, (residual,), started_at, on_sentence)

        complete_text = " ".join(sentences).strip()
        self._print_stream_metrics(started_at, len(sentences))
        if not complete_text:
            return StreamedConversationResponse(
                response=self._unavailable_llm_response(LLMError("OpenRouter returned an empty stream")),
                sentences=(),
                stream_failed=True,
            )

        self._context_store.add_exchange(decision.user_id, decision.text, complete_text, self._clock())
        return StreamedConversationResponse(
            response=self._idle_llm_response(complete_text),
            sentences=tuple(sentences),
        )

    def classify(
        self, text: str, device_id: str, *, user_id: str | None = None,
        received_at: float | None = None,
        interaction_mode: InteractionMode | None = None, low_confidence: bool = False,
    ) -> InteractionDecision:
        mark_latency("classify_start")
        if interaction_mode not in (None, InteractionMode.WAKE, InteractionMode.COMMAND):
            raise ValueError("Modo de interacao invalido")
        if interaction_mode != InteractionMode.COMMAND:
            command = extract_wake_command(text)
            if command:
                technical("[CONVERSATION] wake matched with command")
                text = command
                interaction_mode = InteractionMode.COMMAND
            elif detect_device_state_intent(text, require_wake_word=True) is not None:
                technical("[CONVERSATION] wake matched with device state command")
                interaction_mode = InteractionMode.COMMAND
        conversation_user_id = user_id or device_id
        direct_response, context = self._prepare_response(
            text, conversation_user_id, interaction_mode=interaction_mode, low_confidence=low_confidence,
        )
        mark_latency("classify_done")
        latency_metric("classify_ms", "classify_start", "classify_done")
        if direct_response is None:
            technical("[CONV] classify kind=llm")
            return InteractionDecision(
                InteractionKind.LLM, text, device_id, context=context, user_id=conversation_user_id
            )
        if direct_response.get("interaction_state") == "attention":
            kind = InteractionKind.WAKE if direct_response.get("actions") else InteractionKind.EMPTY
        elif not normalize_text(text):
            kind = InteractionKind.EMPTY
        elif any(action.get("type") in {"show_text", "set_volume", "set_device_state"}
                 for action in direct_response.get("actions", [])):
            kind = InteractionKind.DETERMINISTIC
        else:
            kind = InteractionKind.EMPTY
        technical(f"[CONV] classify kind={kind}")
        return InteractionDecision(
            kind, text, device_id, direct_response=direct_response, user_id=conversation_user_id
        )

    def _prepare_response(
        self, text: str, conversation_user_id: str, *, interaction_mode: InteractionMode | None = None,
        low_confidence: bool = False,
    ) -> tuple[BackendResponse | None, tuple[ConversationMessage, ...]]:
        normalized = normalize_text(text)
        now = self._clock()
        context = self._context_store.get_context(conversation_user_id, now)
        if interaction_mode not in (None, InteractionMode.WAKE, InteractionMode.COMMAND):
            raise ValueError("Modo de interacao invalido")
        if interaction_mode != InteractionMode.COMMAND and is_wake_phrase(text):
            info("Wake: Koda")
            technical("[CONVERSATION] wake matched")
            technical("[CONVERSATION] state=waiting_command")
            technical("[CONVERSATION] attention owner=firmware")
            return (
                {
                    "text": text,
                    "interaction_state": "attention",
                    "actions": [expression("attention")],
                },
                (),
            )

        if interaction_mode != InteractionMode.COMMAND:
            return {"text": text, "interaction_state": "idle", "actions": []}, ()

        technical("[CONVERSATION] state=waiting_command")
        technical(f"[CONVERSATION] command source={'firmware' if interaction_mode is not None else 'legacy'}")
        technical(f'[COMMAND] normalized="{normalized}"')
        if not normalized:
            technical("[CONVERSATION] non-speech transcription -> attention")
            return self._attention_response(text), ()

        if low_confidence:
            technical("[CONVERSATION] low-confidence transcription -> attention")
            return self._attention_response(text), ()

        if normalized in COMMAND_FILLERS or NON_LEXICAL_VOCALIZATION.fullmatch(normalized):
            technical("[CONVERSATION] non-lexical transcription -> attention")
            return self._attention_response(text), ()

        authorization_response = self._resolve_authorization(conversation_user_id, text)
        if authorization_response is not None:
            return authorization_response, ()

        if is_reminder_list_request(normalized):
            return self._reminder_list_response(conversation_user_id)

        cancelled_reminder_text = parse_reminder_cancel_request(normalized)
        if cancelled_reminder_text is not None:
            return self._reminder_cancel_response(conversation_user_id, cancelled_reminder_text)

        reminder_now = self._reminder_now()
        snooze_delay_seconds = parse_reminder_snooze_request(normalized)
        if snooze_delay_seconds is not None:
            return self._reminder_snooze_response(
                conversation_user_id, snooze_delay_seconds, reminder_now.timestamp(),
            )

        reschedule_request = parse_reminder_reschedule_request(normalized, now=reminder_now)
        if reschedule_request is not None:
            return self._reminder_reschedule_response(
                conversation_user_id,
                reschedule_request.text,
                reschedule_request.due_at,
                reminder_now.timestamp(),
            )

        delay_request = parse_reminder_delay_request(normalized)
        if delay_request is not None:
            return self._reminder_delay_response(
                conversation_user_id,
                delay_request.text,
                delay_request.delay_seconds,
                reminder_now.timestamp(),
            )

        reminder_request = parse_reminder_request(normalized, now=reminder_now)
        if reminder_request is not None:
            if self._reminder_service is None:
                return self._reminder_unavailable_response()
            reminder = self._reminder_service.create(
                conversation_user_id,
                reminder_request.text,
                reminder_request.due_at,
                recurrence=reminder_request.recurrence,
                now=reminder_now.timestamp(),
            )
            return (
                {
                    "text": f"Lembrete salvo para {self._format_reminder_due(reminder.due_at)}.",
                    "interaction_state": "idle",
                    "actions": [show_text("Lembrete", "salvo", 2000)],
                },
                (),
            )

        if is_reminder_request(normalized):
            return (
                {
                    "text": "Nao entendi o horario do lembrete.",
                    "interaction_state": "idle",
                    "actions": [show_text("Lembrete", "horario invalido", 2000)],
                },
                (),
            )

        if is_memory_query(normalized):
            return self._memory_list_response(conversation_user_id)

        forgotten_content = parse_memory_forget_request(normalized)
        if forgotten_content is not None:
            return self._memory_forget_response(conversation_user_id, forgotten_content)

        memory_request = parse_explicit_memory_request(normalized)
        if memory_request is not None:
            if contains_sensitive_memory_content(memory_request.content):
                return (
                    {
                        "text": "Nao posso guardar informacoes sensiveis.",
                        "interaction_state": "idle",
                        "actions": [show_text("Memoria", "nao salva", 2000)],
                    },
                    (),
                )
            if self._memory_store is None:
                return (
                    {
                        "text": "Memoria local indisponivel.",
                        "interaction_state": "idle",
                        "actions": [],
                    },
                    (),
                )
            self._memory_store.add(
                conversation_user_id,
                memory_request.category,
                memory_request.content,
                now=self._memory_clock(),
            )
            return (
                {
                    "text": "Memoria salva.",
                    "interaction_state": "idle",
                    "actions": [show_text("Memoria", "salva", 2000)],
                },
                (),
            )

        intent = detect_command_intent(normalized)
        if intent == "sleep":
            technical("[ROUTER] intent=sleep")
            technical("[INTENT] matched=sleep")
            technical("[CONVERSATION] completed -> idle")
            return (
                {
                    "text": "Vou dormir.",
                    "interaction_state": "idle",
                    "actions": [set_device_state("sleeping")],
                },
                (),
            )

        if intent == "wake":
            technical("[ROUTER] intent=wake")
            technical("[INTENT] matched=wake")
            return (
                {
                    "text": "Ja estou acordado.",
                    "interaction_state": "attention",
                    "actions": [],
                },
                (),
            )

        if intent == "hours":
            current_time = get_current_time()
            info("Intenção: hours")
            technical("[ROUTER] intent=hours")
            technical("[INTENT] matched=hours")
            technical(f"[TIME] current={current_time}")
            technical("[CONVERSATION] completed -> idle")
            return (
                {
                    "text": text,
                    "interaction_state": "idle",
                    "actions": [show_text("Agora sao", current_time, 3000)],
                },
                (),
            )

        if intent == "volume":
            volume_percent = extract_volume_percent(normalized)
            if volume_percent is None or volume_percent > 100:
                return (
                    {
                        "text": "O volume deve estar entre 0 e 100%.",
                        "interaction_state": "idle",
                        "actions": [],
                    },
                    (),
                )

            info("Intenção: volume")
            technical("[ROUTER] intent=volume")
            technical(f"[INTENT] matched=volume percent={volume_percent}")
            technical("[CONVERSATION] completed -> idle")
            return (
                {
                    "text": f"Volume ajustado para {volume_percent}%.",
                    "interaction_state": "idle",
                    "actions": [set_volume(volume_percent)],
                },
                (),
            )

        context = self._with_relevant_memories(context, conversation_user_id, text)
        technical("[INTENT] unknown")
        technical("[ROUTER] fallback=llm")
        technical("[CONVERSATION] completed -> idle")
        return None, context

    def _with_relevant_memories(
        self,
        recent_context: tuple[ConversationMessage, ...],
        conversation_user_id: str,
        text: str,
    ) -> tuple[ConversationMessage, ...]:
        if self._memory_store is None:
            return recent_context
        memories = self._memory_store.search(conversation_user_id, text)
        if not memories:
            return recent_context
        memory_context = "\n".join(
            [MEMORY_CONTEXT_HEADER, *(f"- [{memory.category}] {memory.content}" for memory in memories)]
        )
        return (ConversationMessage(role="system", content=memory_context), *recent_context)

    def _resolve_authorization(
        self, user_id: str, text: str,
    ) -> BackendResponse | None:
        if self._action_authorization_session is None:
            return None
        resolution = self._action_authorization_session.resolve(user_id, text)
        if resolution is None:
            return None
        if resolution.status is AuthorizationStatus.APPROVED:
            return self._authorization_response("approved", action=resolution.action)
        if resolution.status is AuthorizationStatus.REJECTED:
            return self._authorization_response("rejected")
        return self._authorization_response("expired")

    @staticmethod
    def _authorization_response(status: str, action: dict[str, object] | None = None) -> BackendResponse:
        messages = {
            "allowed": "Essa acao esta autorizada.",
            "pending": "Posso fazer isso?",
            "denied": "Nao posso executar essa acao.",
            "approved": "Autorizado. A acao ainda nao sera executada.",
            "rejected": "Acao cancelada.",
            "expired": "A autorizacao expirou.",
        }
        response: BackendResponse = {
            "text": messages[status],
            "interaction_state": "idle",
            "actions": [],
        }
        metadata: dict[str, object] = {"authorization": {"status": status}}
        if action is not None:
            metadata["authorization"] = {"status": status, "action": dict(action)}
        response["metadata"] = metadata  # type: ignore[typeddict-unknown-key]
        return response

    def _reminder_list_response(
        self, conversation_user_id: str,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._reminder_service is None:
            return self._reminder_unavailable_response()
        reminders = self._reminder_service.list_pending(conversation_user_id)
        if not reminders:
            return (
                {
                    "text": "Voce nao tem lembretes pendentes.",
                    "interaction_state": "idle",
                    "actions": [show_text("Lembretes", "nenhum", 2000)],
                },
                (),
            )
        summary = "; ".join(
            f"{reminder.text} as {self._format_reminder_due(reminder.due_at)}" for reminder in reminders
        )
        return (
            {
                "text": f"Seus lembretes pendentes: {summary}.",
                "interaction_state": "idle",
                "actions": [show_text("Lembretes", str(len(reminders)), 2000)],
            },
            (),
        )

    def _reminder_cancel_response(
        self, conversation_user_id: str, reminder_text: str,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._reminder_service is None:
            return self._reminder_unavailable_response()
        matches = self._reminder_service.find_pending_matches(conversation_user_id, reminder_text)
        if len(matches) == 1 and self._reminder_service.cancel(
            conversation_user_id, matches[0].id, now=self._reminder_now().timestamp(),
        ):
            return (
                {
                    "text": "Lembrete cancelado.",
                    "interaction_state": "idle",
                    "actions": [show_text("Lembrete", "cancelado", 2000)],
                },
                (),
            )
        if len(matches) > 1:
            message = "Encontrei mais de um lembrete parecido e nao cancelei nenhum."
            display = "ambiguo"
        else:
            message = "Nao encontrei um lembrete pendente para cancelar."
            display = "nao encontrado"
        return (
            {
                "text": message,
                "interaction_state": "idle",
                "actions": [show_text("Lembrete", display, 2000)],
            },
            (),
        )

    def _reminder_reschedule_response(
        self, conversation_user_id: str, reminder_text: str, due_at: float, now: float,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._reminder_service is None:
            return self._reminder_unavailable_response()
        matches = self._reminder_service.find_pending_matches(conversation_user_id, reminder_text)
        if len(matches) == 1:
            updated = self._reminder_service.reschedule(
                conversation_user_id, matches[0].id, due_at, now=now,
            )
            if updated is not None:
                return (
                    {
                        "text": f"Lembrete reagendado para {self._format_reminder_due(updated.due_at)}.",
                        "interaction_state": "idle",
                        "actions": [show_text("Lembrete", "reagendado", 2000)],
                    },
                    (),
                )
        return self._reminder_match_failure_response("reagendar", matches)

    def _reminder_delay_response(
        self, conversation_user_id: str, reminder_text: str, delay_seconds: float, now: float,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._reminder_service is None:
            return self._reminder_unavailable_response()
        matches = self._reminder_service.find_pending_matches(conversation_user_id, reminder_text)
        if len(matches) == 1:
            updated = self._reminder_service.snooze(
                conversation_user_id, matches[0].id, delay_seconds, now=now,
            )
            if updated is not None:
                return (
                    {
                        "text": f"Lembrete adiado para {self._format_reminder_due(updated.due_at)}.",
                        "interaction_state": "idle",
                        "actions": [show_text("Lembrete", "adiado", 2000)],
                    },
                    (),
                )
        return self._reminder_match_failure_response("adiar", matches)

    def _reminder_snooze_response(
        self, conversation_user_id: str, delay_seconds: float, now: float,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._reminder_service is None:
            return self._reminder_unavailable_response()
        candidates = self._reminder_service.find_delivered_for_snooze(conversation_user_id)
        if len(candidates) == 1:
            updated = self._reminder_service.snooze(
                conversation_user_id, candidates[0].id, delay_seconds, now=now,
            )
            if updated is not None:
                return (
                    {
                        "text": f"Lembrete adiado para {self._format_reminder_due(updated.due_at)}.",
                        "interaction_state": "idle",
                        "actions": [show_text("Lembrete", "adiado", 2000)],
                    },
                    (),
                )
        if len(candidates) > 1:
            message = "Encontrei mais de um lembrete entregue e nao adiei nenhum."
            display = "ambiguo"
        else:
            message = "Nao tenho um lembrete entregue associado a este pedido."
            display = "nao associado"
        return (
            {
                "text": message,
                "interaction_state": "idle",
                "actions": [show_text("Lembrete", display, 2000)],
            },
            (),
        )

    @staticmethod
    def _reminder_match_failure_response(
        operation: str, matches: tuple[object, ...],
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if len(matches) > 1:
            message = f"Encontrei mais de um lembrete parecido e nao consegui {operation} nenhum."
            display = "ambiguo"
        else:
            message = f"Nao encontrei um lembrete pendente para {operation}."
            display = "nao encontrado"
        return (
            {
                "text": message,
                "interaction_state": "idle",
                "actions": [show_text("Lembrete", display, 2000)],
            },
            (),
        )

    @staticmethod
    def _reminder_unavailable_response() -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        return (
            {
                "text": "Lembretes locais indisponiveis.",
                "interaction_state": "idle",
                "actions": [],
            },
            (),
        )

    @staticmethod
    def _format_reminder_due(due_at: float) -> str:
        return datetime.fromtimestamp(due_at, REMINDER_TIMEZONE).strftime("%H:%M")

    def _memory_list_response(
        self, conversation_user_id: str,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._memory_store is None:
            return self._memory_unavailable_response()
        memories = self._memory_store.list(conversation_user_id)
        if not memories:
            return (
                {
                    "text": "Ainda nao tenho memorias salvas sobre voce.",
                    "interaction_state": "idle",
                    "actions": [show_text("Memorias", "nenhuma", 2000)],
                },
                (),
            )
        visible_memories = memories[:MAX_MEMORIES_IN_LIST_RESPONSE]
        summary = "; ".join(memory.content for memory in visible_memories)
        remainder = len(memories) - len(visible_memories)
        suffix = f" Tenho mais {remainder} memoria(s) salva(s)." if remainder else ""
        return (
            {
                "text": f"Eu lembro: {summary}.{suffix}",
                "interaction_state": "idle",
                "actions": [show_text("Memorias", str(len(memories)), 2000)],
            },
            (),
        )

    def _memory_forget_response(
        self, conversation_user_id: str, content: str,
    ) -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        if self._memory_store is None:
            return self._memory_unavailable_response()
        normalized_content = normalized_memory_content(content)
        matches = tuple(
            memory for memory in self._memory_store.list(conversation_user_id)
            if normalized_memory_content(memory.content) == normalized_content
        )
        if len(matches) == 1 and self._memory_store.delete(conversation_user_id, matches[0].id):
            return (
                {
                    "text": "Memoria esquecida.",
                    "interaction_state": "idle",
                    "actions": [show_text("Memoria", "esquecida", 2000)],
                },
                (),
            )
        if len(matches) > 1:
            message = "Encontrei mais de uma memoria igual e nao apaguei nenhuma."
            display = "ambigua"
        else:
            message = "Nao encontrei uma memoria exata para esquecer."
            display = "nao encontrada"
        return (
            {
                "text": message,
                "interaction_state": "idle",
                "actions": [show_text("Memoria", display, 2000)],
            },
            (),
        )

    @staticmethod
    def _memory_unavailable_response() -> tuple[BackendResponse, tuple[ConversationMessage, ...]]:
        return (
            {
                "text": "Memoria local indisponivel.",
                "interaction_state": "idle",
                "actions": [],
            },
            (),
        )

    @staticmethod
    def _attention_response(text: str) -> BackendResponse:
        return {"text": text, "interaction_state": "attention", "actions": []}

    @staticmethod
    def _idle_llm_response(text: str) -> BackendResponse:
        return {"text": text, "interaction_state": "idle", "actions": []}

    @staticmethod
    def _append_sentences(
        sentences: list[str],
        new_sentences: tuple[str, ...],
        started_at: float,
        on_sentence: Callable[[str, int, int], None] | None,
    ) -> None:
        for sentence in new_sentences:
            sequence = len(sentences)
            if sequence == 0:
                mark_latency("first_sentence")
                latency_metric("first_sentence_ms", "llm_request_start", "first_sentence")
            sentences.append(sentence)
            if on_sentence is not None:
                on_sentence(sentence, sequence, round((time.monotonic() - started_at) * 1000))

    @staticmethod
    def _print_stream_metrics(started_at: float, sentence_count: int) -> None:
        technical(f"[RT] llm_complete_ms={round((time.monotonic() - started_at) * 1000)}")
        technical(f"[RT] sentences={sentence_count}")

    def _unavailable_llm_response(self, error: LLMError) -> BackendResponse:
        log_error("Falha ao acessar o modelo de linguagem.")
        technical(f"[LLM] provider unavailable error={type(error).__name__}")
        technical("[LLM] provider fallback disabled")
        return self._idle_llm_response(LLM_UNAVAILABLE_MESSAGE)
