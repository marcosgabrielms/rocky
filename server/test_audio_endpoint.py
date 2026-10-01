"""Testes determinísticos do endpoint WAV destinado ao ESP32."""

import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

import app


class AudioEndpointTest(TestCase):
    def test_serves_current_pcm_wav(self) -> None:
        with TemporaryDirectory() as directory:
            audio_path = Path(directory) / "rocky_response.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(32000)
                wav_file.writeframes(b"\x00\x00" * 320)

            with patch.object(app, "TTS_OUTPUT_PATH", audio_path):
                response = TestClient(app.app).get("/audio/rocky_response.wav")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content[:4], b"RIFF")

    def test_returns_not_found_without_generated_audio(self) -> None:
        with TemporaryDirectory() as directory:
            missing_path = Path(directory) / "rocky_response.wav"
            with patch.object(app, "TTS_OUTPUT_PATH", missing_path):
                response = TestClient(app.app).get("/audio/rocky_response.wav")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    import unittest

    unittest.main()
