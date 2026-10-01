"""Estado conversacional em memória, separado por dispositivo Rocky."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from models.actions import BackendResponse, expression, idle_response, show_text
from services.commands import detect_command_intent, get_current_time, normalize_text
from services.llm import ConversationMessage, LLMClient, LLMError, SentenceBuffer, StreamingLLMClient


WAKE_PHRASES = frozenset({"ok rocky", "rocky"})
MAX_CONTEXT_MESSAGES = 6
CONTEXT_TTL_SECONDS = 5 * 60
LLM_UNAVAILABLE_MESSAGE = "Não consegui acessar meu modelo de linguagem agora."
LLM_INTERRUPTED_MESSAGE = "Não consegui concluir minha resposta."


def is_wake_phrase(text: str) -> bool:
    normalized = "".join(character if character.isalnum() else " " for character in text.casefold())
    return " ".join(normalized.split()) in WAKE_PHRASES


@dataclass
class ConversationHistory:
    messages: list[ConversationMessage]
    last_interaction_at: float


class InteractionKind(StrEnum):
    EMPTY = "empty"
    WAKE = "wake"
    DETERMINISTIC = "deterministic"
    LLM = "llm"


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
        self._command_window_seconds = command_window_ms / 1000
        self._command_window_ms = command_window_ms
        self._deadlines: dict[str, float] = {}
        self._histories: dict[str, ConversationHistory] = {}
        self._llm_client = llm_client
        self._clock = clock

    def has_active_command_window(self, device_id: str) -> bool:
        deadline = self._deadlines.get(device_id)
        return deadline is not None and self._clock() <= deadline

    def build_response(self, text: str, device_id: str) -> BackendResponse:
        return self.build_response_from_decision(self.classify(text, device_id))

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
        buffer = SentenceBuffer()
        sentences: list[str] = []
        received_text = False
        print("[RT] llm_request_start")
        try:
            for chunk in self._llm_client.stream_response(decision.text, context=decision.context):
                if not chunk:
                    continue
                now = time.monotonic()
                if first_token_at is None:
                    first_token_at = now
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

    def classify(self, text: str, device_id: str) -> InteractionDecision:
        direct_response, context = self._prepare_response(text, device_id)
        if direct_response is None:
            print("[CONV] classify kind=llm")
            return InteractionDecision(InteractionKind.LLM, text, device_id, context=context)
        if is_wake_phrase(text):
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
        self, text: str, device_id: str
    ) -> tuple[BackendResponse | None, tuple[ConversationMessage, ...]]:
        normalized = normalize_text(text)
        now = self._clock()
        context = self._get_context(device_id, now)
        if is_wake_phrase(text):
            self._deadlines[device_id] = now + self._command_window_seconds
            print("[CONVERSATION] wake matched")
            print("[CONVERSATION] state=waiting_command")
            print(f"[CONVERSATION] deadline started={self._command_window_ms}ms")
            return (
                {
                    "text": text,
                    "interaction_state": "attention",
                    "command_window_ms": self._command_window_ms,
                    "actions": [expression("attention")],
                },
                (),
            )

        deadline = self._deadlines.get(device_id)
        if deadline is None:
            return {"text": text, "interaction_state": "idle", "actions": []}, ()

        remaining_ms = int((deadline - now) * 1000)
        print(f"[CONVERSATION] remaining_ms={remaining_ms}")
        if now > deadline:
            self._deadlines.pop(device_id, None)
            print("[CONVERSATION] timeout -> idle")
            return idle_response(text), ()

        print("[CONVERSATION] state=waiting_command")
        print(f'[COMMAND] normalized="{normalized}"')
        self._deadlines.pop(device_id, None)
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
