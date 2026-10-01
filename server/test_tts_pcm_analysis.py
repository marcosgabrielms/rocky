"""Testes determinísticos das métricas PCM do WAV final do Rocky."""

import struct
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from services.tts.response_speaker import analyze_pcm_wav


class PcmAnalysisTest(TestCase):
    def test_reports_peak_rms_clipping_and_high_samples(self) -> None:
        samples = (0, 1000, -1000, 32767, -32768, 30000)
        with TemporaryDirectory() as directory:
            wav_path = Path(directory) / "metrics.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(32000)
                wav_file.writeframes(struct.pack("<6h", *samples))

            metrics = analyze_pcm_wav(wav_path)

        self.assertEqual(metrics.minimum, -32768)
        self.assertEqual(metrics.maximum, 32767)
        self.assertEqual(metrics.peak_absolute, 32768)
        self.assertAlmostEqual(metrics.mean, 29999 / 6)
        self.assertEqual(metrics.negative_clipping, 1)
        self.assertEqual(metrics.positive_clipping, 1)
        self.assertEqual(metrics.sample_count, 6)
        self.assertEqual(metrics.above_90_percent, 3)
        self.assertEqual(metrics.above_95_percent, 2)
        self.assertEqual(metrics.above_99_percent, 2)
