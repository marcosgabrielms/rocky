"""Adaptador local do Piper TTS."""

import wave
from pathlib import Path

from piper import PiperVoice


class PiperTTS:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._voice: PiperVoice | None = None

    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        if not text.strip():
            raise ValueError("O texto para síntese não pode estar vazio.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wav_file:
            self._get_voice().synthesize_wav(text, wav_file)

    def _get_voice(self) -> PiperVoice:
        if self._voice is None:
            self._voice = PiperVoice.load(self._model_path)

        return self._voice
