"""Transcrição in-memory com faster-whisper."""

import io
import wave
from functools import lru_cache

from fastapi import HTTPException
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
NO_SPEECH_PROBABILITY_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
STT_DIAGNOSTICS = False


@lru_cache
def get_model() -> WhisperModel:
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def report_transcription_diagnostics(
    segments: list[object], duration: float, duration_after_vad: float, accepted: bool
) -> None:
    if not STT_DIAGNOSTICS:
        return

    print(f"[STT DIAG] duration={duration:.2f}")
    print(f"[STT DIAG] duration_after_vad={duration_after_vad:.2f}")
    print(f"[STT DIAG] segments={len(segments)}")
    for index, segment in enumerate(segments):
        print(f'[STT DIAG] segment={index} text="{segment.text}"')
        print(f"[STT DIAG] segment={index} no_speech_prob={segment.no_speech_prob:.4f}")
        print(f"[STT DIAG] segment={index} avg_logprob={segment.avg_logprob:.4f}")
        print(f"[STT DIAG] segment={index} compression_ratio={segment.compression_ratio:.4f}")
    print(f"[STT DIAG] decision={'accepted' if accepted else 'rejected'}")


def get_confident_text(segments: list[object], duration: float, duration_after_vad: float) -> str:
    text = "".join(segment.text for segment in segments).strip()
    all_low_confidence = bool(segments) and all(
        segment.no_speech_prob > NO_SPEECH_PROBABILITY_THRESHOLD
        and segment.avg_logprob < AVG_LOGPROB_THRESHOLD
        for segment in segments
    )
    accepted = bool(segments) and duration_after_vad > 0 and bool(text) and not all_low_confidence
    report_transcription_diagnostics(segments, duration, duration_after_vad, accepted)

    return text if accepted else ""


def transcribe(audio_data: bytes, filename: str | None, use_wake_hotword: bool) -> str:
    audio_buffer = io.BytesIO(audio_data)
    audio_buffer.name = filename or "rocky.wav"
    hotwords = "Rocky" if use_wake_hotword else None
    try:
        segments, info = get_model().transcribe(
            audio_buffer,
            language="pt",
            hotwords=hotwords,
            vad_filter=True,
        )
        return get_confident_text(list(segments), info.duration, info.duration_after_vad)
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail="Não foi possível transcrever o áudio.") from error
    finally:
        audio_buffer.close()


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
