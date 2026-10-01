"""Síntese incremental local de frases prontas para o pipeline realtime."""

import time
import wave
from dataclasses import dataclass
from collections.abc import Collection
from pathlib import Path
from threading import Lock

from services.tts.base import TTSClient
from services.tts.response_speaker import SUPPORTED_SAMPLE_RATE, resample_wav


@dataclass(frozen=True)
class RealtimeAudioSegment:
    """Artefato WAV gerado para uma sentença completa."""

    sequence: int
    text: str
    audio_path: Path
    sample_rate: int
    channels: int
    bits_per_sample: int
    duration_ms: int
    audio_bytes: int


class RealtimeTTSSession:
    """Mantém ordem e nomes exclusivos dos segmentos de uma interação."""

    def __init__(self, tts: "RealtimeTTS", interaction_id: int) -> None:
        self._tts = tts
        self._interaction_id = interaction_id

    def synthesize(self, text: str, sequence: int) -> RealtimeAudioSegment:
        return self._tts._synthesize(self._interaction_id, text, sequence)


class RealtimeTTS:
    """Converte cada sentença finalizada em um WAV PCM independente e ordenado."""

    _FILENAME_PREFIX = "rocky_rt_"

    def __init__(self, tts_client: TTSClient, output_directory: Path) -> None:
        self._tts_client = tts_client
        self._output_directory = output_directory
        self._lock = Lock()
        self._next_interaction_id = 0

    @property
    def output_directory(self) -> Path:
        return self._output_directory

    def begin_interaction(
        self,
        interaction_id: int | None = None,
        protected_audio_paths: Collection[Path] = (),
    ) -> RealtimeTTSSession:
        """Remove somente artefatos realtime anteriores antes de uma nova interação."""
        with self._lock:
            self._output_directory.mkdir(parents=True, exist_ok=True)
            protected_paths = {path.resolve() for path in protected_audio_paths}
            for audio_path in self._output_directory.glob(f"{self._FILENAME_PREFIX}*.wav"):
                if audio_path.resolve() not in protected_paths:
                    audio_path.unlink()
            if interaction_id is None:
                interaction_id = self._next_interaction_id
            self._next_interaction_id = max(self._next_interaction_id, interaction_id + 1)
        return RealtimeTTSSession(self, interaction_id)

    def _synthesize(self, interaction_id: int, text: str, sequence: int) -> RealtimeAudioSegment:
        if not text.strip():
            raise ValueError("A sentença realtime não pode estar vazia.")

        output_path = self._output_directory / f"{self._FILENAME_PREFIX}{interaction_id:04d}_{sequence:03d}.wav"
        started_at = time.monotonic()
        print(f"[RT] tts_start seq={sequence}")
        with self._lock:
            self._tts_client.synthesize_to_wav(text, output_path)
            resample_wav(output_path, SUPPORTED_SAMPLE_RATE)

        elapsed_ms = round((time.monotonic() - started_at) * 1000)
        segment = self._read_segment(output_path, text, sequence)
        print(f"[RT] tts_ready seq={sequence} ms={elapsed_ms}")
        print(f"[RT] tts_duration_ms={segment.duration_ms}")
        print(f"[RT] audio_bytes={segment.audio_bytes}")
        return segment

    @staticmethod
    def _read_segment(output_path: Path, text: str, sequence: int) -> RealtimeAudioSegment:
        with wave.open(str(output_path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise ValueError("O WAV realtime deve usar PCM sem compressão.")
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()

        if channels != 1 or sample_width != 2 or sample_rate != SUPPORTED_SAMPLE_RATE:
            raise ValueError("O WAV realtime deve ser mono PCM16 em 32000 Hz.")

        return RealtimeAudioSegment(
            sequence=sequence,
            text=text,
            audio_path=output_path,
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=sample_width * 8,
            duration_ms=round(frame_count * 1000 / sample_rate),
            audio_bytes=output_path.stat().st_size,
        )
