"""Síntese incremental local de frases prontas para o pipeline realtime."""

import time
import wave
from dataclasses import dataclass
from collections.abc import Collection
from pathlib import Path
from threading import Lock

from services.tts.base import TTSClient
from services.tts.response_speaker import SUPPORTED_SAMPLE_RATE, resample_wav
from services.latency import latency_metric, mark_latency

ORPHAN_AUDIO_MAXIMUM_AGE_SECONDS = 300


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
        """Abre a sessao sem apagar arquivos de outro produtor ainda em andamento."""
        with self._lock:
            self._output_directory.mkdir(parents=True, exist_ok=True)
            if interaction_id is None:
                interaction_id = self._next_interaction_id
            self._next_interaction_id = max(self._next_interaction_id, interaction_id + 1)
        self.cleanup_orphaned_audio(protected_audio_paths)
        return RealtimeTTSSession(self, interaction_id)

    def cleanup_orphaned_audio(self, protected_audio_paths: Collection[Path] = ()) -> None:
        """Remove WAVs antigos sem fila, preservando arquivos recentes e paths externos."""
        protected = {path.resolve() for path in protected_audio_paths}
        now = time.time()  # mtime de arquivos usa epoch, nao o relogio monotonic da fila.
        with self._lock:
            for path in self._output_directory.glob(f"{self._FILENAME_PREFIX}*.wav"):
                resolved = path.resolve()
                if resolved.parent != self._output_directory.resolve() or resolved in protected:
                    continue
                if now - path.stat().st_mtime >= ORPHAN_AUDIO_MAXIMUM_AGE_SECONDS:
                    path.unlink()

    def _synthesize(self, interaction_id: int, text: str, sequence: int) -> RealtimeAudioSegment:
        if not text.strip():
            raise ValueError("A sentença realtime não pode estar vazia.")

        output_path = self._output_directory / f"{self._FILENAME_PREFIX}{interaction_id:04d}_{sequence:03d}.wav"
        started_at = time.monotonic()
        print(f"[RT] tts_start seq={sequence}")
        if sequence == 0:
            mark_latency("tts_start")
        with self._lock:
            try:
                self._tts_client.synthesize_to_wav(text, output_path)
                resample_wav(output_path, SUPPORTED_SAMPLE_RATE)
                segment = self._read_segment(output_path, text, sequence)
            except Exception:
                if output_path.resolve().parent == self._output_directory.resolve():
                    output_path.unlink(missing_ok=True)
                raise

        elapsed_ms = round((time.monotonic() - started_at) * 1000)
        if sequence == 0:
            mark_latency("tts_ready")
            latency_metric("tts_first_sentence_ms", "tts_start", "tts_ready")
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
