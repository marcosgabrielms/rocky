"""Estado conversacional em memória, separado por dispositivo Rocky."""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from models.actions import BackendResponse, expression, idle_response, show_text
from services.commands import detect_command_intent, get_current_time, normalize_text
from services.llm import ConversationMessage, LLMClient, LLMError, SentenceBuffer, StreamingLLMClient
from services.latency import latency_metric, mark_latency


WAKE_PREFIX = re.compile(r"^\W*koda(?=$|\W)(?P<command>.*)$", re.IGNORECASE | re.DOTALL)
MAX_CONTEXT_MESSAGES = 6
CONTEXT_TTL_SECONDS = 5 * 60
LLM_UNAVAILABLE_MESSAGE = "Não consegui acessar meu modelo de linguagem agora."
LLM_INTERRUPTED_MESSAGE = "Não consegui concluir minha resposta."


def is_wake_phrase(text: str) -> bool:
    return extract_wake_command(text) == ""


def extract_wake_command(text: str) -> str | None:
    """Somente Koda como primeiro token; None indica ausencia de wake valida."""
    match = WAKE_PREFIX.match(text)
    if match is None:
        return None
    return re.sub(r"^\W+", "", match.group("command")).strip()


@dataclass
class ConversationHistory:
    messages: list[ConversationMessage]
    last_interaction_at: float


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


@dataclass(frozen=True)
class StreamedConversationResponse:
    """Resposta final e sentenças prontas para uma futura saída em tempo real."""

    response: BackendResponse
    sentences: tuple[str, ...]
    stream_failed: bool = False


class ConversationManager:
    def __init__(
        self, command_window_ms: int, llm_client: LLMClient, clock: Callable[[], float] = time.monotonic
    ) -> None:
        # Argumento legado; somente o corpo fisico controla a espera por fala.
        del command_window_ms
        self._waiting_commands: set[str] = set()
        self._histories: dict[str, ConversationHistory] = {}
        self._llm_client = llm_client
        self._clock = clock

    def capture_request_time(self) -> float:
        """Timestamp funcional em segundos, independente do tracker de diagnóstico."""
        return self._clock()

    def has_active_command_window(self, device_id: str, *, received_at: float | None = None) -> bool:
        """Compatibilidade para clientes sem modo explicito, sem deadline no backend."""
        return device_id in self._waiting_commands

    def build_response(
        self, text: str, device_id: str, *, received_at: float | None = None,
        interaction_mode: InteractionMode | None = None,
    ) -> BackendResponse:
        return self.build_response_from_decision(
            self.classify(text, device_id, received_at=received_at, interaction_mode=interaction_mode)
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
            self._store_exchange(decision.device_id, decision.text, result.text, self._clock())

        return self._idle_llm_response(result.text)

    def build_stream_response(
        self,
        text: str,
        device_id: str,
        on_sentence: Callable[[str, int, int], None] | None = None,
    ) -> StreamedConversationResponse:
        """Executa uma única geração em streaming sem mudar o contrato HTTP atual."""
        return self.build_stream_response_from_decision(self.classify(text, device_id), on_sentence)

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
        print("[RT] llm_request_start")
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
                    print(f"[RT] llm_first_token_ms={round((now - started_at) * 1000)}")
                received_text = True
                new_sentences = buffer.push(chunk)
                if new_sentences and first_sentence_at is None:
                    first_sentence_at = now
                    print(f"[RT] first_sentence_ms={round((now - started_at) * 1000)}")
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
                print(f"[RT] first_sentence_ms={round((time.monotonic() - started_at) * 1000)}")
            self._append_sentences(sentences, (residual,), started_at, on_sentence)

        complete_text = " ".join(sentences).strip()
        self._print_stream_metrics(started_at, len(sentences))
        if not complete_text:
            return StreamedConversationResponse(
                response=self._unavailable_llm_response(LLMError("OpenRouter returned an empty stream")),
                sentences=(),
                stream_failed=True,
            )

        self._store_exchange(decision.device_id, decision.text, complete_text, self._clock())
        return StreamedConversationResponse(
            response=self._idle_llm_response(complete_text),
            sentences=tuple(sentences),
        )

    def classify(
        self, text: str, device_id: str, *, received_at: float | None = None,
        interaction_mode: InteractionMode | None = None,
    ) -> InteractionDecision:
        mark_latency("classify_start")
        if interaction_mode not in (None, InteractionMode.WAKE, InteractionMode.COMMAND):
            raise ValueError("Modo de interacao invalido")
        if interaction_mode != InteractionMode.COMMAND:
            command = extract_wake_command(text)
            if command:
                print("[CONVERSATION] wake matched with command")
                text = command
                interaction_mode = InteractionMode.COMMAND
        direct_response, context = self._prepare_response(text, device_id, interaction_mode=interaction_mode)
        mark_latency("classify_done")
        latency_metric("classify_ms", "classify_start", "classify_done")
        if direct_response is None:
            print("[CONV] classify kind=llm")
            return InteractionDecision(InteractionKind.LLM, text, device_id, context=context)
        if direct_response.get("interaction_state") == "attention":
            kind = InteractionKind.WAKE
        elif not normalize_text(text):
            kind = InteractionKind.EMPTY
        elif any(action.get("type") == "show_text" for action in direct_response.get("actions", [])):
            kind = InteractionKind.DETERMINISTIC
        else:
            kind = InteractionKind.EMPTY
        print(f"[CONV] classify kind={kind}")
        return InteractionDecision(kind, text, device_id, direct_response=direct_response)

    def _prepare_response(
        self, text: str, device_id: str, *, interaction_mode: InteractionMode | None = None
    ) -> tuple[BackendResponse | None, tuple[ConversationMessage, ...]]:
        normalized = normalize_text(text)
        now = self._clock()
        context = self._get_context(device_id, now)
        if interaction_mode not in (None, InteractionMode.WAKE, InteractionMode.COMMAND):
            raise ValueError("Modo de interacao invalido")
        if interaction_mode == InteractionMode.WAKE:
            self._waiting_commands.discard(device_id)
        if interaction_mode != InteractionMode.COMMAND and is_wake_phrase(text):
            self._waiting_commands.add(device_id)
            print("[CONVERSATION] wake matched")
            print("[CONVERSATION] state=waiting_command")
            print("[CONVERSATION] attention owner=firmware")
            return (
                {
                    "text": text,
                    "interaction_state": "attention",
                    "actions": [expression("attention")],
                },
                (),
            )

        is_command = interaction_mode == InteractionMode.COMMAND or (
            interaction_mode is None and device_id in self._waiting_commands
        )
        if not is_command:
            return {"text": text, "interaction_state": "idle", "actions": []}, ()

        print("[CONVERSATION] state=waiting_command")
        print(f"[CONVERSATION] command source={'firmware' if interaction_mode is not None else 'legacy'}")
        print(f'[COMMAND] normalized="{normalized}"')
        self._waiting_commands.discard(device_id)
        if not normalized:
            print("[CONVERSATION] empty transcription -> idle")
            return idle_response(text), ()

        if detect_command_intent(normalized) == "hours":
            current_time = get_current_time()
            print("[ROUTER] intent=hours")
            print("[INTENT] matched=hours")
            print(f"[TIME] current={current_time}")
            print("[CONVERSATION] completed -> idle")
            return (
                {
                    "text": text,
                    "interaction_state": "idle",
                    "actions": [show_text("Agora sao", current_time, 3000)],
                },
                (),
            )

        print("[INTENT] unknown")
        print("[ROUTER] fallback=llm")
        print("[CONVERSATION] completed -> idle")
        return None, context

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
        print(f"[RT] llm_complete_ms={round((time.monotonic() - started_at) * 1000)}")
        print(f"[RT] sentences={sentence_count}")

    def _unavailable_llm_response(self, error: LLMError) -> BackendResponse:
        print(f"[LLM] free model unavailable error={error}")
        print("[LLM] paid fallback disabled")
        return self._idle_llm_response(LLM_UNAVAILABLE_MESSAGE)

    def _get_context(self, device_id: str, now: float) -> tuple[ConversationMessage, ...]:
        history = self._histories.get(device_id)
        if history is None:
            return ()

        if now - history.last_interaction_at >= CONTEXT_TTL_SECONDS:
            self._histories.pop(device_id, None)
            return ()

        return tuple(history.messages)

    def _store_exchange(self, device_id: str, user_text: str, assistant_text: str, now: float) -> None:
        history = self._histories.get(device_id)
        messages = history.messages if history is not None else []
        messages.extend(
            (
                ConversationMessage(role="user", content=user_text),
                ConversationMessage(role="assistant", content=assistant_text),
            )
        )
        self._histories[device_id] = ConversationHistory(
            messages=messages[-MAX_CONTEXT_MESSAGES:],
            last_interaction_at=now,
        )
