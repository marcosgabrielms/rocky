"""Orquestra a síntese e a reprodução local de respostas do Rocky."""

import winsound
from pathlib import Path
from threading import Lock

from models.actions import BackendResponse
from services.tts.base import TTSClient


def get_spoken_text(response: BackendResponse, request_text: str) -> str | None:
    """Seleciona apenas o conteúdo destinado ao usuário para reprodução local."""
    for action in response.get("actions", []):
        if action["type"] == "show_text":
            return " ".join((action["line1"], action["line2"])).strip() or None

    response_text = response.get("text", "").strip()
    if not response_text or response_text == request_text.strip():
        return None

    return response_text


class LocalResponseSpeaker:
    def __init__(self, tts_client: TTSClient, output_path: Path) -> None:
        self._tts_client = tts_client
        self._output_path = output_path
        self._lock = Lock()

    def speak_response(self, response: BackendResponse, request_text: str) -> bool:
        spoken_text = get_spoken_text(response, request_text)
        if spoken_text is None:
            return False

        try:
            with self._lock:
                self._tts_client.synthesize_to_wav(spoken_text, self._output_path)
                winsound.PlaySound(str(self._output_path), winsound.SND_FILENAME)
        except Exception as error:
            print(f"[TTS] error={error}")
            return False

        print(f'[TTS] played text="{spoken_text}"')
        return True
