"""Endpoints HTTP e composição dos serviços do backend Rocky."""

import csv
import io
import math
import os
import struct
import time
import wave
from pathlib import Path
from threading import Thread

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from services.conversation import ConversationManager, InteractionKind
from services.llm import OpenRouterLLMClient
from services.realtime_audio import RealtimeAudioService
from services.stt import SAMPLE_RATE, transcribe as transcribe_audio, validate_wav
from services.tts import LocalResponseSpeaker, PiperTTS, RealtimeTTS, get_local_playback_enabled
from services.audio_queue import AudioProductionStatus


MAX_UPLOAD_BYTES = 1024 * 1024
EXPECTED_CONTENT_TYPE = "audio/wav"
COMMAND_WINDOW_MS = 10000
DEFAULT_CONVERSATION_LLM_MODEL = "openrouter/free"
DATASET_DIRECTORY = Path(__file__).parent / "dataset"
CALIBRATION_DIRECTORY = Path(__file__).parent / "calibration"
TTS_DIRECTORY = Path(__file__).parent / "local" / "tts"
TTS_MODEL_PATH = TTS_DIRECTORY / "voices" / "pt_BR-jeff-medium.onnx"
TTS_OUTPUT_PATH = TTS_DIRECTORY / "rocky_response.wav"
REALTIME_TTS_DIRECTORY = TTS_DIRECTORY / "realtime"
REALTIME_LONG_POLL_MAX_MS = 1000
REALTIME_LONG_POLL_INTERVAL_MS = 50

app = FastAPI()
llm_client = OpenRouterLLMClient(model=os.getenv("OPENROUTER_MODEL", DEFAULT_CONVERSATION_LLM_MODEL))
conversation_manager = ConversationManager(COMMAND_WINDOW_MS, llm_client)
response_speaker = LocalResponseSpeaker(
    PiperTTS(TTS_MODEL_PATH),
    TTS_OUTPUT_PATH,
    enable_local_playback=get_local_playback_enabled(),
)
realtime_audio_service = RealtimeAudioService(
    conversation_manager,
    RealtimeTTS(PiperTTS(TTS_MODEL_PATH), REALTIME_TTS_DIRECTORY),
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile | None = File(default=None),
    x_rocky_device: str = Header(default="rocky-01"),
) -> dict[str, object]:
    audio_data = await read_wav_upload(file)
    duration_seconds = validate_wav(audio_data)
    print("[STT] request received")
    print(f"[STT] wav: {SAMPLE_RATE} Hz mono 16-bit duration={duration_seconds:.1f}s")

    use_wake_hotword = not conversation_manager.has_active_command_window(x_rocky_device)
    print(f"[STT] mode={'wake' if use_wake_hotword else 'command'}")
    text = transcribe_audio(audio_data, file.filename, use_wake_hotword)
    print("[STT] transcription completed")
    response = conversation_manager.build_response(text, x_rocky_device)
    response["audio_available"] = response_speaker.speak_response(response, text)
    return response


@app.post("/transcribe/realtime")
async def transcribe_realtime(
    file: UploadFile | None = File(default=None),
    x_rocky_device: str = Header(default="rocky-01"),
) -> dict[str, object]:
    audio_data = await read_wav_upload(file)
    duration_seconds = validate_wav(audio_data)
    print("[RTD] request received")
    print(f"[RTD] wav duration={duration_seconds:.1f}s")
    use_wake_hotword = not conversation_manager.has_active_command_window(x_rocky_device)
    text = transcribe_audio(audio_data, file.filename, use_wake_hotword)
    decision = conversation_manager.classify(text, x_rocky_device)
    print(f"[RTD] classify kind={decision.kind}")
    if decision.kind is not InteractionKind.LLM:
        response = conversation_manager.build_response_from_decision(decision)
        response["audio_available"] = response_speaker.speak_response(response, text)
        return {"realtime": False, "interaction_id": None, **response}

    interaction_id = realtime_audio_service.create_interaction()
    print(f"[RTD] interaction_created id={interaction_id}")
    started_at = time.monotonic()

    def produce() -> None:
        result = realtime_audio_service.build_stream_audio_response_from_decision(decision, interaction_id)
        if result.audio_segments:
            elapsed_ms = round((time.monotonic() - started_at) * 1000)
            print(f"[RTD] first_audio_available_ms={elapsed_ms}")

    Thread(target=produce, name=f"rocky-realtime-{interaction_id}", daemon=True).start()
    return {
        "realtime": True,
        "interaction_id": interaction_id,
        "text": "",
        "interaction_state": "thinking",
        "actions": [],
        "audio_available": False,
    }


@app.get("/realtime/audio/{interaction_id}/next")
def get_next_realtime_audio(interaction_id: int, timeout_ms: int = REALTIME_LONG_POLL_MAX_MS):
    timeout_ms = min(max(timeout_ms, 0), REALTIME_LONG_POLL_MAX_MS)
    started_at = time.monotonic()
    queue = realtime_audio_service.audio_queue
    while True:
        try:
            item = queue.next_item(interaction_id)
            state = queue.state(interaction_id)
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Interacao realtime desconhecida.") from error

        if item is not None:
            print(f"[RTD] served id={interaction_id} seq={item.sequence}")
            return FileResponse(
                item.audio_path,
                media_type="audio/wav",
                headers={
                    "X-Rocky-Interaction": str(item.interaction_id),
                    "X-Rocky-Sequence": str(item.sequence),
                    "X-Rocky-Status": "ready",
                    "X-Rocky-Duration-Ms": str(item.duration_ms),
                    "X-Rocky-Sample-Rate": str(item.sample_rate),
                    "X-Rocky-Channels": str(item.channels),
                    "X-Rocky-Bits-Per-Sample": str(item.bits_per_sample),
                    "X-Rocky-Audio-Bytes": str(item.audio_bytes),
                },
            )
        if state.production_status is AudioProductionStatus.COMPLETED:
            print(f"[RTD] done id={interaction_id}")
            return _realtime_status("done")
        if state.production_status is AudioProductionStatus.FAILED:
            return _realtime_status("failed")
        if state.production_status is AudioProductionStatus.CANCELLED:
            return _realtime_status("cancelled")
        if (time.monotonic() - started_at) * 1000 >= timeout_ms:
            print(f"[RTD] next_wait_ms={round((time.monotonic() - started_at) * 1000)}")
            return _realtime_status("pending")
        time.sleep(REALTIME_LONG_POLL_INTERVAL_MS / 1000)


@app.post("/realtime/audio/{interaction_id}/{sequence}/consumed")
def acknowledge_realtime_audio(interaction_id: int, sequence: int) -> dict[str, int | str]:
    try:
        realtime_audio_service.audio_queue.mark_consumed(interaction_id, sequence)
    except ValueError as error:
        raise HTTPException(status_code=409, detail="ACK realtime invalido.") from error
    print(f"[RTD] consumed id={interaction_id} seq={sequence}")
    return {"status": "consumed", "interaction_id": interaction_id, "sequence": sequence}


def _realtime_status(status: str) -> JSONResponse:
    return JSONResponse({"status": status})


@app.get("/audio/rocky_response.wav")
def get_response_audio() -> FileResponse:
    if not TTS_OUTPUT_PATH.is_file() or TTS_OUTPUT_PATH.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Audio indisponivel.")

    print(f"[AUDIO] served path={TTS_OUTPUT_PATH.name} bytes={TTS_OUTPUT_PATH.stat().st_size}")
    return FileResponse(TTS_OUTPUT_PATH, media_type="audio/wav")


@app.post("/dataset")
async def save_dataset(file: UploadFile | None = File(default=None), label: str = "") -> dict[str, int | str]:
    if file is None:
        raise HTTPException(status_code=400, detail="Arquivo WAV ausente.")
    if label != "rocky":
        raise HTTPException(status_code=400, detail="Label invalido.")

    audio_data = await read_wav_upload(file)
    duration_seconds = validate_wav(audio_data)
    destination = save_unique_dataset_file(label, audio_data)
    index = int(destination.stem.rsplit("_", 1)[1])
    print(f"[DATASET] label={label}")
    print(f"[DATASET] saved={destination.name}")
    print(f"[DATASET] duration={duration_seconds:.2f}s")
    return {"label": label, "index": index, "filename": destination.name}


@app.post("/calibration/audio")
async def save_calibration_audio(
    file: UploadFile | None = File(default=None),
    mode: str = Form(),
    distance_cm: int = Form(),
    sample_id: int = Form(),
    voice_level: str = Form(default="normal"),
    fan_state: str = Form(default="off"),
    noise_rms: int = Form(default=0),
    noise_peak_rms: int = Form(default=0),
    raw24_rms: int = Form(default=0),
    pcm16_rms: int = Form(default=0),
    peak_abs: int = Form(default=0),
    clipping: int = Form(default=0),
    vad_trigger_rms: int = Form(default=0),
    vad_trigger_delay_ms: int = Form(default=0),
) -> dict[str, int | float | str]:
    if file is None:
        raise HTTPException(status_code=400, detail="Arquivo WAV ausente.")
    validate_calibration_metadata(mode, distance_cm, sample_id, voice_level, fan_state)
    audio_data = await read_wav_upload(file)
    duration_seconds = validate_wav(audio_data)
    pcm_metrics = calculate_pcm16_metrics(audio_data)
    destination = save_calibration_file(mode, distance_cm, voice_level, fan_state, sample_id, audio_data)
    snr_estimate_db = calculate_snr_estimate(raw24_rms, noise_rms)
    append_calibration_result({
        "sample_id": sample_id,
        "mode": mode,
        "distance_cm": distance_cm,
        "voice_level": voice_level,
        "fan_state": fan_state,
        "noise_rms": noise_rms,
        "raw24_rms": raw24_rms,
        "pcm16_rms": pcm16_rms,
        "speech_rms": raw24_rms,
        "peak_abs": peak_abs,
        "snr_estimate_db": snr_estimate_db,
        "vad_trigger_rms": vad_trigger_rms,
        "vad_trigger_delay_ms": vad_trigger_delay_ms,
        "duration_ms": round(duration_seconds * 1000),
        "clipping": clipping,
        "wav_file": destination.name,
    })
    print(f"[CAL] saved={destination.name}")
    print(f"[CAL] pcm16_rms={pcm_metrics['rms']} peak_abs={pcm_metrics['peak_abs']}")
    return {
        "filename": destination.name,
        "duration_ms": round(duration_seconds * 1000),
        "pcm16_rms": pcm_metrics["rms"],
        "peak_abs": pcm_metrics["peak_abs"],
    }


async def read_wav_upload(file: UploadFile | None) -> bytes:
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
    return audio_data


def validate_calibration_metadata(
    mode: str,
    distance_cm: int,
    sample_id: int,
    voice_level: str,
    fan_state: str,
) -> None:
    if mode not in {"raw", "vad", "noise"}:
        raise HTTPException(status_code=400, detail="Modo de calibracao invalido.")
    if distance_cm not in {20, 40, 60, 80}:
        raise HTTPException(status_code=400, detail="Distancia de calibracao invalida.")
    if voice_level not in {"low", "normal", "high"}:
        raise HTTPException(status_code=400, detail="Nivel de voz invalido.")
    if fan_state not in {"on", "off"}:
        raise HTTPException(status_code=400, detail="Estado do ventilador invalido.")
    if sample_id == 0:
        raise HTTPException(status_code=400, detail="Identificador da amostra invalido.")


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


def save_calibration_file(
    mode: str,
    distance_cm: int,
    voice_level: str,
    fan_state: str,
    sample_id: int,
    audio_data: bytes,
) -> Path:
    CALIBRATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if mode == "noise":
        filename = f"noise_{distance_cm}cm_fan{fan_state}_{sample_id:03d}.wav"
    else:
        filename = f"{mode}_{distance_cm}cm_{voice_level}_fan{fan_state}_{sample_id:03d}.wav"
    destination = CALIBRATION_DIRECTORY / filename
    with destination.open("xb") as output_file:
        output_file.write(audio_data)
    return destination


def calculate_pcm16_metrics(audio_data: bytes) -> dict[str, int]:
    with wave.open(io.BytesIO(audio_data), "rb") as wav_file:
        samples = tuple(sample[0] for sample in struct.iter_unpack("<h", wav_file.readframes(wav_file.getnframes())))

    peak_absolute = max(abs(sample) for sample in samples)
    rms = int(math.sqrt(sum(sample * sample for sample in samples) / len(samples)))
    return {"rms": rms, "peak_abs": peak_absolute}


def calculate_snr_estimate(speech_rms: int, noise_rms: int) -> float:
    if speech_rms == 0 or noise_rms == 0:
        return 0.0
    return round(20.0 * math.log10(speech_rms / noise_rms), 1)


def append_calibration_result(result: dict[str, int | float | str]) -> None:
    CALIBRATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = CALIBRATION_DIRECTORY / "results.csv"
    write_header = not destination.exists()
    with destination.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=result.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(result)
