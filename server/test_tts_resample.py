"""Testes determinísticos da conversão do WAV do Piper para 32 kHz."""

import struct
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from services.tts.response_speaker import SUPPORTED_SAMPLE_RATE, resample_wav


SOURCE_SAMPLE_RATE = 22050
SOURCE_FRAME_COUNT = SOURCE_SAMPLE_RATE


class TtsResampleTest(TestCase):
    def test_resample_preserves_duration_and_pcm_format(self) -> None:
        with TemporaryDirectory() as directory:
            wav_path = Path(directory) / "source.wav"
            source_samples = (index % 2000 - 1000 for index in range(SOURCE_FRAME_COUNT))
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SOURCE_SAMPLE_RATE)
                wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in source_samples))

            resample_wav(wav_path, SUPPORTED_SAMPLE_RATE)

            with wave.open(str(wav_path), "rb") as wav_file:
                output_frames = wav_file.getnframes()
                output_duration = output_frames / wav_file.getframerate()
                self.assertEqual(wav_file.getcomptype(), "NONE")
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.getframerate(), SUPPORTED_SAMPLE_RATE)

        expected_frames = SOURCE_FRAME_COUNT * SUPPORTED_SAMPLE_RATE / SOURCE_SAMPLE_RATE
        self.assertLessEqual(abs(output_frames - expected_frames), 2)
        self.assertAlmostEqual(output_duration, 1.0, places=3)


if __name__ == "__main__":
    import unittest

    unittest.main()
