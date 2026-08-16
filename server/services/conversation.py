"""Estado conversacional em memória, separado por dispositivo Rocky."""

import time
from collections.abc import Callable
from dataclasses import dataclass

from models.actions import BackendResponse, expression, idle_response, show_text
from services.commands import detect_command_intent, get_current_time, normalize_text
from services.llm import ConversationMessage, LLMClient, LLMError


WAKE_PHRASES = frozenset({"ok rocky", "rocky"})
MAX_CONTEXT_MESSAGES = 6
CONTEXT_TTL_SECONDS = 5 * 60


def is_wake_phrase(text: str) -> bool:
    normalized = "".join(character if character.isalnum() else " " for character in text.casefold())
    return " ".join(normalized.split()) in WAKE_PHRASES


@dataclass
class ConversationHistory:
    messages: list[ConversationMessage]
    last_interaction_at: float


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
        normalized = normalize_text(text)
        now = self._clock()
        self._get_context(device_id, now)
        if is_wake_phrase(text):
            self._deadlines[device_id] = now + self._command_window_seconds
            print("[CONVERSATION] wake matched")
            print("[CONVERSATION] state=waiting_command")
            print(f"[CONVERSATION] deadline started={self._command_window_ms}ms")
            return {
                "text": text,
                "interaction_state": "attention",
                "command_window_ms": self._command_window_ms,
                "actions": [expression("attention")],
            }

        deadline = self._deadlines.get(device_id)
        if deadline is None:
            return {"text": text, "interaction_state": "idle", "actions": []}

        remaining_ms = int((deadline - now) * 1000)
        print(f"[CONVERSATION] remaining_ms={remaining_ms}")
        if now > deadline:
            self._deadlines.pop(device_id, None)
            print("[CONVERSATION] timeout -> idle")
            return idle_response(text)

        print("[CONVERSATION] state=waiting_command")
        print(f'[COMMAND] normalized="{normalized}"')
        self._deadlines.pop(device_id, None)
        if not normalized:
            print("[CONVERSATION] empty transcription -> idle")
            return idle_response(text)

        if detect_command_intent(normalized) == "hours":
            current_time = get_current_time()
            print("[ROUTER] intent=hours")
            print("[INTENT] matched=hours")
            print(f"[TIME] current={current_time}")
            print("[CONVERSATION] completed -> idle")
            return {
                "text": text,
                "interaction_state": "idle",
                "actions": [show_text("Agora sao", current_time, 3000)],
            }

        print("[INTENT] unknown")
        print("[ROUTER] fallback=llm")
        print("[CONVERSATION] completed -> idle")
        try:
            context = self._get_context(device_id, now)
            result = self._llm_client.ask(text, context=context)
        except LLMError as error:
            print(f"[LLM] fallback error={error}")
            return {
                "text": "Nao consegui responder agora.",
                "interaction_state": "idle",
                "actions": [],
            }

        if result.text.strip():
            self._store_exchange(device_id, text, result.text, self._clock())

        return {
            "text": result.text,
            "interaction_state": "idle",
            "actions": [],
        }

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
