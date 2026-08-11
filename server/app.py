import io
import time
import wave
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from faster_whisper import WhisperModel


MAX_UPLOAD_BYTES = 1024 * 1024
EXPECTED_CONTENT_TYPE = "audio/wav"
SAMPLE_RATE = 16000
SAMPLE_WIDTH_BYTES = 2
CHANNELS = 1
MODEL_SIZE = "small"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

app = FastAPI()
DATASET_DIRECTORY = Path(__file__).parent / "dataset"
COMMAND_WINDOW_SECONDS = 10.0
COMMAND_WINDOW_MS = int(COMMAND_WINDOW_SECONDS * 1000)
SAO_PAULO_TIMEZONE = ZoneInfo("America/Sao_Paulo")
HOURS_ALIASES = frozenset({"horas", "hora", "hores", "oras", "ors"})
conversation_deadlines: dict[str, float] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile | None = File(default=None),
    x_rocky_device: str = Header(default="rocky-01"),
) -> dict[str, object]:
    if file is None:
        raise HTTPException(status_code=400, detail="Arquivo WAV ausente.")

    if file.content_type != EXPECTED_CONTENT_TYPE:
        raise HTTPException(status_code=415, detail="O arquivo deve ter o tipo audio/wav.")

    audio_data = await file.read()
    await file.close()

    if not audio_data:
        raise HTTPException(status_code=400, detail="Arquivo WAV vazio.")

    if len(audio_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo WAV excede o limite de 1 MB.")

    duration_seconds = validate_wav(audio_data)
    print("[STT] request received")
    print(f"[STT] wav: {SAMPLE_RATE} Hz mono 16-bit duration={duration_seconds:.1f}s")

    hotwords = "Rocky" if not has_active_command_window(x_rocky_device) else None
    print(f"[STT] mode={'wake' if hotwords else 'command'}")
    audio_buffer = io.BytesIO(audio_data)
    audio_buffer.name = file.filename or "rocky.wav"

    try:
        segments, _ = get_model().transcribe(
            audio_buffer,
            language="pt",
            hotwords=hotwords,
        )
        text = "".join(segment.text for segment in segments).strip()
    except RuntimeError:
        raise HTTPException(status_code=502, detail="Não foi possível transcrever o áudio.")
    finally:
        audio_buffer.close()

    print("[STT] transcription completed")
    return build_transcription_response(text, x_rocky_device)


@app.post("/dataset")
async def save_dataset(file: UploadFile | None = File(default=None), label: str = "") -> dict[str, int | str]:
    if file is None:
        raise HTTPException(status_code=400, detail="Arquivo WAV ausente.")
    if label != "rocky":
        raise HTTPException(status_code=400, detail="Label invalido.")
    if file.content_type != EXPECTED_CONTENT_TYPE:
        raise HTTPException(status_code=415, detail="O arquivo deve ter o tipo audio/wav.")
    audio_data = await file.read()
    await file.close()
    if not audio_data:
        raise HTTPException(status_code=400, detail="Arquivo WAV vazio.")
    if len(audio_data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo WAV excede o limite de 1 MB.")
    duration_seconds = validate_wav(audio_data)
    destination = save_unique_dataset_file(label, audio_data)
    index = int(destination.stem.rsplit("_", 1)[1])
    print(f"[DATASET] label={label}")
    print(f"[DATASET] saved={destination.name}")
    print(f"[DATASET] duration={duration_seconds:.2f}s")
    return {"label": label, "index": index, "filename": destination.name}


def save_unique_dataset_file(label: str, audio_data: bytes) -> Path:
    directory = DATASET_DIRECTORY / label
    directory.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        destination = directory / f"{label}_{index:04d}.wav"
        try:
            with destination.open("xb") as output_file:
                output_file.write(audio_data)
            return destination
        except FileExistsError:
            index += 1


def build_transcription_response(text: str, device_id: str) -> dict[str, object]:
    normalized = text.strip().casefold().rstrip(".?!,;:")
    now = time.monotonic()
    if normalized == "rocky":
        deadline = now + COMMAND_WINDOW_SECONDS
        conversation_deadlines[device_id] = deadline
        print("[CONVERSATION] wake matched")
        print("[CONVERSATION] state=waiting_command")
        print(f"[CONVERSATION] deadline started={COMMAND_WINDOW_MS}ms")
        return {
            "text": text,
            "interaction_state": "attention",
            "command_window_ms": COMMAND_WINDOW_MS,
            "actions": [{"type": "expression", "value": "attention"}],
        }

    deadline = conversation_deadlines.get(device_id)
    if deadline is None:
        return {"text": text, "interaction_state": "idle", "actions": []}

    remaining_ms = int((deadline - now) * 1000)
    print(f"[CONVERSATION] remaining_ms={remaining_ms}")
    if now > deadline:
        conversation_deadlines.pop(device_id, None)
        print("[CONVERSATION] timeout -> idle")
        return idle_response(text)

    print("[CONVERSATION] state=waiting_command")
    print(f"[COMMAND] normalized=\"{normalized}\"")
    conversation_deadlines.pop(device_id, None)
    intent = detect_command_intent(normalized)
    if intent == "hours":
        current_time = datetime.now(SAO_PAULO_TIMEZONE).strftime("%H:%M")
        print("[INTENT] matched=hours")
        print(f"[TIME] current={current_time}")
        print("[CONVERSATION] completed -> idle")
        return {
            "text": text,
            "interaction_state": "idle",
            "actions": [{
                "type": "show_text",
                "line1": "Agora sao",
                "line2": current_time,
                "duration_ms": 3000,
            }],
        }

    print("[INTENT] unknown")
    print("[CONVERSATION] completed -> idle")
    return idle_response(text)


def detect_command_intent(normalized_text: str) -> str | None:
    if normalized_text in HOURS_ALIASES:
        return "hours"
    return None


def has_active_command_window(device_id: str) -> bool:
    deadline = conversation_deadlines.get(device_id)
    return deadline is not None and time.monotonic() <= deadline


def idle_response(text: str) -> dict[str, object]:
    return {
        "text": text,
        "interaction_state": "idle",
        "actions": [{"type": "expression", "value": "idle"}],
    }

    return {"text": text, "interaction_state": "idle", "actions": []}


@lru_cache
def get_model() -> WhisperModel:
    return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


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
    except (EOFError, wave.Error):
        raise HTTPException(status_code=400, detail="WAV inválido.")
