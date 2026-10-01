"""Testes determinísticos da decisão de fala local do Rocky."""

import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from models.actions import BackendResponse, expression, show_text
from services.tts.response_speaker import LocalResponseSpeaker, get_spoken_text


class FakeTTSClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        self.calls.append((text, output_path))
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 22050)


class FailingTTSClient:
    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        raise RuntimeError("falha simulada")


class ResponseSpeakerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.output_path = Path(self._temporary_directory.name) / "rocky_response.wav"
        self.tts_client = FakeTTSClient()
        self.speaker = LocalResponseSpeaker(self.tts_client, self.output_path, enable_local_playback=False)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_llm_response_is_synthesized(self) -> None:
        response: BackendResponse = {"text": "Olá. Como posso ajudar?", "interaction_state": "idle", "actions": []}

        spoken = self.speaker.speak_response(response, "Qual é a sua função?")

        self.assertTrue(spoken)
        self.assertEqual(self.tts_client.calls, [("Olá. Como posso ajudar?", self.output_path)])
        with wave.open(str(self.output_path), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 32000)

    def test_deterministic_response_with_text_action_is_synthesized(self) -> None:
        response: BackendResponse = {
            "text": "horas",
            "interaction_state": "idle",
            "actions": [show_text("Agora sao", "12:00", 3000)],
        }

        spoken = self.speaker.speak_response(response, "horas")

        self.assertTrue(spoken)
        self.assertEqual(self.tts_client.calls, [("Agora sao 12:00", self.output_path)])

    def test_wake_is_not_synthesized(self) -> None:
        response: BackendResponse = {
            "text": "Ok, Rocky",
            "interaction_state": "attention",
            "actions": [expression("attention")],
        }

        self.assertFalse(self.speaker.speak_response(response, "Ok, Rocky"))
        self.assertEqual(self.tts_client.calls, [])

    def test_empty_text_is_not_synthesized(self) -> None:
        response: BackendResponse = {"text": "", "interaction_state": "idle", "actions": []}

        self.assertFalse(self.speaker.speak_response(response, ""))
        self.assertEqual(self.tts_client.calls, [])

    def test_tts_failure_preserves_response(self) -> None:
        response: BackendResponse = {"text": "Resposta válida", "interaction_state": "idle", "actions": []}
        speaker = LocalResponseSpeaker(FailingTTSClient(), self.output_path)

        self.assertFalse(speaker.speak_response(response, "Pergunta"))
        self.assertEqual(response, {"text": "Resposta válida", "interaction_state": "idle", "actions": []})

    def test_internal_transcription_echo_is_not_synthesized(self) -> None:
        response: BackendResponse = {"text": "Pergunta em idle", "interaction_state": "idle", "actions": []}

        self.assertIsNone(get_spoken_text(response, "Pergunta em idle"))

if __name__ == "__main__":
    unittest.main()
