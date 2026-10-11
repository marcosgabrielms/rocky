"""Transcrição in-memory com faster-whisper."""

import io
import wave
from dataclasses import dataclass
from functools import lru_cache

from fastapi import HTTPException
from faster_whisper import WhisperModel

from services.latency import latency_metric, mark_latency
from services.runtime_log import technical


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
NO_SPEECH_PROBABILITY_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
STT_DIAGNOSTICS = False
# Baseline explícito da API 1.2.1; mudar apenas após comparação de latência/qualidade.
BEAM_SIZE = 5
BEST_OF = 5
CONDITION_ON_PREVIOUS_TEXT = True


@dataclass(frozen=True)
class TranscriptionResult:
    """Texto reconhecido e o sinal minimo de confianca do Whisper."""

    text: str
    has_speech: bool
    low_confidence: bool

    @property
    def accepted_text(self) -> str:
        return self.text if self.has_speech and not self.low_confidence else ""


@lru_cache
def get_model() -> WhisperModel:
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def report_transcription_diagnostics(
    segments: list[object], duration: float, duration_after_vad: float, accepted: bool
) -> None:
    if not STT_DIAGNOSTICS:
        return

    technical(f"[STT DIAG] duration={duration:.2f}")
    technical(f"[STT DIAG] duration_after_vad={duration_after_vad:.2f}")
    technical(f"[STT DIAG] segments={len(segments)}")
    for index, segment in enumerate(segments):
        technical(f'[STT DIAG] segment={index} text="{segment.text}"')
        technical(f"[STT DIAG] segment={index} no_speech_prob={segment.no_speech_prob:.4f}")
        technical(f"[STT DIAG] segment={index} avg_logprob={segment.avg_logprob:.4f}")
        technical(f"[STT DIAG] segment={index} compression_ratio={segment.compression_ratio:.4f}")
    technical(f"[STT DIAG] decision={'accepted' if accepted else 'rejected'}")


def assess_transcription(
    segments: list[object], duration: float, duration_after_vad: float
) -> TranscriptionResult:
    text = "".join(segment.text for segment in segments).strip()
    all_low_confidence = bool(segments) and all(
        segment.no_speech_prob > NO_SPEECH_PROBABILITY_THRESHOLD
        and segment.avg_logprob < AVG_LOGPROB_THRESHOLD
        for segment in segments
    )
    has_speech = bool(segments) and duration_after_vad > 0 and bool(text)
    accepted = has_speech and not all_low_confidence
    report_transcription_diagnostics(segments, duration, duration_after_vad, accepted)
    return TranscriptionResult(text=text if has_speech else "", has_speech=has_speech,
                               low_confidence=all_low_confidence)


def get_confident_text(segments: list[object], duration: float, duration_after_vad: float) -> str:
    return assess_transcription(segments, duration, duration_after_vad).accepted_text


def transcribe_with_quality(
    audio_data: bytes, filename: str | None, use_wake_hotword: bool
) -> TranscriptionResult:
    audio_buffer = io.BytesIO(audio_data)
    audio_buffer.name = filename or "rocky.wav"
    hotwords = "Koda" if use_wake_hotword else None
    try:
        mark_latency("model_load_start")
        model = get_model()
        mark_latency("model_load_done")
        latency_metric("model_load_ms", "model_load_start", "model_load_done")
        segments, info = model.transcribe(
            audio_buffer,
            language="pt",
            hotwords=hotwords,
            vad_filter=True,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
            word_timestamps=False,
            initial_prompt=None,
        )
        return assess_transcription(list(segments), info.duration, info.duration_after_vad)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail="Não foi possível transcrever o áudio.") from error
    finally:
        audio_buffer.close()


def transcribe(audio_data: bytes, filename: str | None, use_wake_hotword: bool) -> str:
    return transcribe_with_quality(audio_data, filename, use_wake_hotword).accepted_text


def validate_wav(audio_data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
            if wav_file.getnchannels() != CHANNELS:
                raise HTTPException(status_code=400, detail="O WAV deve ser mono.")
            if wav_file.getsampwidth() != SAMPLE_WIDTH_BYTES:
                raise HTTPException(status_code=400, detail="O WAV deve usar PCM16.")
            if wav_file.getframerate() != SAMPLE_RATE:
                raise HTTPException(status_code=400, detail="O WAV deve usar 16000 Hz.")

            frames = wav_file.getnframes()
            if frames == 0:
                raise HTTPException(status_code=400, detail="O WAV não possui amostras.")
            return frames / SAMPLE_RATE
    except (EOFError, wave.Error) as error:
        raise HTTPException(status_code=400, detail="WAV inválido.") from error
