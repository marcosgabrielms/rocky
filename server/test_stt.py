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

    def transcribe_with_quality(
        self, segments: list[SimpleNamespace], duration_after_vad: float
    ) -> tuple[stt.TranscriptionResult, FakeModel]:
        model = FakeModel(segments, duration_after_vad)
        with patch("services.stt.get_model", return_value=model):
            result = stt.transcribe_with_quality(b"audio", "rocky.wav", use_wake_hotword=False)
        return result, model

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

    def test_preserves_low_confidence_signal_for_post_wake_routing(self) -> None:
        result, _ = self.transcribe_with_quality([segment("e", 0.7, -1.1)], duration_after_vad=1.0)

        self.assertEqual(result.text, "e")
        self.assertTrue(result.has_speech)
        self.assertTrue(result.low_confidence)
        self.assertEqual(result.accepted_text, "")

    def test_preserves_confident_short_command(self) -> None:
        result, _ = self.transcribe_with_quality([segment("hora", 0.1, -0.2)], duration_after_vad=1.0)

        self.assertEqual(result.text, "hora")
        self.assertFalse(result.low_confidence)
        self.assertEqual(result.accepted_text, "hora")

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
        self.assertEqual(model.calls[0]["hotwords"], "Koda")
        self.assertEqual(model.calls[0]["beam_size"], 5)
        self.assertEqual(model.calls[0]["best_of"], 5)
        self.assertTrue(model.calls[0]["condition_on_previous_text"])
        self.assertFalse(model.calls[0]["word_timestamps"])
        self.assertIsNone(model.calls[0]["initial_prompt"])

    def test_command_mode_has_no_wake_hint(self) -> None:
        model = FakeModel([segment("horas", 0.1, -0.2)], 1.0)
        with patch("services.stt.get_model", return_value=model):
            text = stt.transcribe(b"audio", None, use_wake_hotword=False)
        self.assertEqual(text, "horas")
        self.assertIsNone(model.calls[0]["hotwords"])


if __name__ == "__main__":
    unittest.main()
