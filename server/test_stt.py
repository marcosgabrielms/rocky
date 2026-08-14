"""Testes determinísticos para o filtro de confiança do STT."""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import sys

try:
    from fastapi import HTTPException
except ModuleNotFoundError:
    fastapi_module = ModuleType("fastapi")

    class HTTPException(Exception):
        pass

    fastapi_module.HTTPException = HTTPException
    sys.modules["fastapi"] = fastapi_module

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError:
    faster_whisper_module = ModuleType("faster_whisper")

    class WhisperModel:
        pass

    faster_whisper_module.WhisperModel = WhisperModel
    sys.modules["faster_whisper"] = faster_whisper_module

from services import stt


class FakeModel:
    def __init__(self, segments: list[SimpleNamespace], duration_after_vad: float) -> None:
        self._segments = segments
        self._info = SimpleNamespace(duration=1.0, duration_after_vad=duration_after_vad)
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio: object, **kwargs: object) -> tuple[iter, SimpleNamespace]:
        self.calls.append(kwargs)
        return iter(self._segments), self._info


def segment(text: str, no_speech_prob: float, avg_logprob: float) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        no_speech_prob=no_speech_prob,
        avg_logprob=avg_logprob,
        compression_ratio=1.0,
    )


class SttConfidenceTest(unittest.TestCase):
    def transcribe_with(self, segments: list[SimpleNamespace], duration_after_vad: float) -> tuple[str, FakeModel]:
        model = FakeModel(segments, duration_after_vad)
        with patch("services.stt.get_model", return_value=model):
            text = stt.transcribe(b"audio", "rocky.wav", use_wake_hotword=True)
        return text, model

    def test_rejects_empty_segments(self) -> None:
        text, _ = self.transcribe_with([], duration_after_vad=1.0)

        self.assertEqual(text, "")

    def test_rejects_empty_text(self) -> None:
        text, _ = self.transcribe_with([segment("   ", 0.1, -0.2)], duration_after_vad=1.0)

        self.assertEqual(text, "")

    def test_rejects_audio_removed_by_vad(self) -> None:
        text, _ = self.transcribe_with([segment("Rocky", 0.1, -0.2)], duration_after_vad=0.0)

        self.assertEqual(text, "")

    def test_rejects_only_low_confidence_segments(self) -> None:
        text, _ = self.transcribe_with(
            [
                segment("Ok", 0.7, -1.1),
                segment(" Legendas", 0.8, -1.2),
            ],
            duration_after_vad=1.0,
        )

        self.assertEqual(text, "")

    def test_preserves_text_with_a_confident_segment(self) -> None:
        text, model = self.transcribe_with(
            [
                segment("Rocky", 0.2, -0.3),
                segment(" horas", 0.8, -1.2),
            ],
            duration_after_vad=1.0,
        )

        self.assertEqual(text, "Rocky horas")
        self.assertTrue(model.calls[0]["vad_filter"])
        self.assertEqual(model.calls[0]["language"], "pt")
        self.assertEqual(model.calls[0]["hotwords"], "Rocky")


if __name__ == "__main__":
    unittest.main()
