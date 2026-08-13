"""Estado conversacional em memória, separado por dispositivo Rocky."""

import time

from models.actions import BackendResponse, expression, idle_response, show_text
from services.commands import detect_command_intent, get_current_time, normalize_text


class ConversationManager:
    def __init__(self, command_window_ms: int) -> None:
        self._command_window_seconds = command_window_ms / 1000
        self._command_window_ms = command_window_ms
        self._deadlines: dict[str, float] = {}

    def has_active_command_window(self, device_id: str) -> bool:
        deadline = self._deadlines.get(device_id)
        return deadline is not None and time.monotonic() <= deadline

    def build_response(self, text: str, device_id: str) -> BackendResponse:
        normalized = normalize_text(text)
        now = time.monotonic()
        if normalized == "rocky":
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
        if detect_command_intent(normalized) == "hours":
            current_time = get_current_time()
            print("[INTENT] matched=hours")
            print(f"[TIME] current={current_time}")
            print("[CONVERSATION] completed -> idle")
            return {
                "text": text,
                "interaction_state": "idle",
                "actions": [show_text("Agora sao", current_time, 3000)],
            }

        print("[INTENT] unknown")
        print("[CONVERSATION] completed -> idle")
        return idle_response(text)
