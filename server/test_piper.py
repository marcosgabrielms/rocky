"""Um unico modelo lazy, compartilhado com sintese serializada; sem Piper real."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from services.tts.piper import PiperTTS


class PiperSharingTest(unittest.TestCase):
    def test_shared_client_loads_once_for_concurrent_syntheses(self) -> None:
        voice = Mock()

        def synthesize(text, wav):
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\x00\x00" * 20)

        voice.synthesize_wav.side_effect = synthesize
        voice_class = Mock()
        voice_class.load.return_value = voice
        with TemporaryDirectory() as directory, patch("services.tts.piper.PiperVoice", voice_class):
            client = PiperTTS(Path(directory) / "voice.onnx")
            voice_class.load.assert_not_called()
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda index: client.synthesize_to_wav("Resposta.", Path(directory) / f"{index}.wav"), range(4)))
        voice_class.load.assert_called_once()
        self.assertEqual(voice.synthesize_wav.call_count, 4)

    def test_app_shares_adapter_between_sync_and_realtime(self) -> None:
        import app
        self.assertIs(app.response_speaker._tts_client, app.realtime_audio_service._realtime_tts._tts_client)


if __name__ == "__main__":
    unittest.main()
