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


@lru_cache
def get_model() -> WhisperModel:
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe(audio_data: bytes, filename: str | None, use_wake_hotword: bool) -> str:
    audio_buffer = io.BytesIO(audio_data)
    audio_buffer.name = filename or "rocky.wav"
    hotwords = "Rocky" if use_wake_hotword else None
    try:
        segments, _ = get_model().transcribe(
            audio_buffer,
            language="pt",
            hotwords=hotwords,
        )
        return "".join(segment.text for segment in segments).strip()
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
