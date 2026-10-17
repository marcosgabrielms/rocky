"""Endpoints HTTP e composição dos serviços do backend Rocky."""

import csv
import io
import math
import os
import struct
import time
import wave
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from threading import Thread
from contextvars import copy_context

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from koda_core import CoreDecision, InteractionResult
from services.commands import normalize_text
from services.conversation import InteractionKind, InteractionMode, extract_wake_command, is_wake_phrase
from services.llm import ChatGPTPlanLLMClient
from services.realtime_audio import RealtimeAudioService
from services.stt import SAMPLE_RATE, TranscriptionResult, transcribe_with_quality as transcribe_audio, validate_wav
from services.tts import LocalResponseSpeaker, PiperTTS, RealtimeTTS, get_local_playback_enabled
from services.audio_queue import AudioProductionStatus
from services.latency import latency_metric, latency_scope, mark_latency, report_latency, report_latency_summary
from services.runtime_log import error as log_error, info, technical
from services.tts.response_speaker import get_spoken_text
from runtime import KodaRuntime


MAX_UPLOAD_BYTES = 1024 * 1024
EXPECTED_CONTENT_TYPE = "audio/wav"
DATASET_DIRECTORY = Path(__file__).parent / "dataset"
CALIBRATION_DIRECTORY = Path(__file__).parent / "calibration"
TTS_DIRECTORY = Path(__file__).parent / "local" / "tts"
TTS_MODEL_PATH = TTS_DIRECTORY / "voices" / "pt_BR-jeff-medium.onnx"
TTS_OUTPUT_PATH = TTS_DIRECTORY / "rocky_response.wav"
REALTIME_TTS_DIRECTORY = TTS_DIRECTORY / "realtime"
REALTIME_LONG_POLL_MAX_MS = 1000
REALTIME_LONG_POLL_INTERVAL_MS = 50
LOCAL_USER_ID = "local-user"

llm_client = ChatGPTPlanLLMClient()
runtime = KodaRuntime.create(llm_client)


@asynccontextmanager
async def _runtime_lifespan(_: FastAPI):
    """Mantem o scheduler local ativo somente durante a vida da API."""
    runtime.reminder_scheduler.start()
    try:
        yield
    finally:
        runtime.reminder_scheduler.stop()


app = FastAPI(lifespan=_runtime_lifespan)
conversation_manager = runtime.conversation_manager
conversation_core_adapter = runtime.conversation_core_adapter
command_session = runtime.command_session
interaction_service = runtime.interaction_service
tts_client = PiperTTS(TTS_MODEL_PATH)
response_speaker = LocalResponseSpeaker(
    tts_client,
    TTS_OUTPUT_PATH,
    enable_local_playback=get_local_playback_enabled(),
)
realtime_audio_service = RealtimeAudioService(
    conversation_manager,
    RealtimeTTS(tts_client, REALTIME_TTS_DIRECTORY),
)


class TextInteractionPayload(BaseModel):
    """Entrada textual de uma interface que jÃ¡ possui texto."""

    text: str
    device_id: str


def _conversation_input(
    transcription: TranscriptionResult | str, use_wake_hotword: bool
) -> tuple[str, bool]:
    """Mantem mocks legados e so entrega texto de baixa confianca apos a wake."""
    if isinstance(transcription, str):
        return transcription, False
    text = transcription.accepted_text if use_wake_hotword else transcription.text
    return text, transcription.low_confidence


def _report_transcription_and_response(
    text: str,
    low_confidence: bool,
    response: dict[str, object],
    kind: InteractionKind | None = None,
) -> None:
    """Mostra apenas a conversa útil; descartes continuam disponíveis no verbose."""
    if low_confidence or not text or response.get("interaction_state") == "attention":
        return

    spoken_text = get_spoken_text(response, text)
    if kind is InteractionKind.EMPTY or (spoken_text is None and response.get("text") == text):
        return

    info(f'Ouvi: "{text}"')
    if spoken_text:
        info(f'Resposta: "{spoken_text}"')


def _interaction_result_to_http_response(result: InteractionResult) -> dict[str, object]:
    """Expõe somente o contrato HTTP já consumido pelo firmware."""
    return {
        "text": result.text,
        "interaction_state": result.interaction_state,
        "actions": [dict(action) for action in result.actions],
    }


def _interaction_result_to_text_http_response(result: InteractionResult) -> dict[str, object]:
    """ExpÃµe o resultado textual completo sem acionar efeitos do robÃ´."""
    response = _interaction_result_to_http_response(result)
    if result.provider is not None:
        response["provider"] = result.provider
    if result.model is not None:
        response["model"] = result.model
    if result.usage is not None:
        response["usage"] = asdict(result.usage)
    if result.metadata:
        response["metadata"] = dict(result.metadata)
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/interact")
def interact(payload: TextInteractionPayload) -> dict[str, object]:
    """Processa texto pronto sem depender do pipeline de Ã¡udio do robÃ´."""
    text = payload.text.strip()
    device_id = payload.device_id.strip()
    if not text or not device_id:
        raise HTTPException(status_code=422, detail="text e device_id devem ser preenchidos.")

    core_decision = interaction_service.classify(
        user_id=LOCAL_USER_ID,
        device_id=device_id,
        input_type="text",
        text=text,
        interaction_mode=InteractionMode.COMMAND,
        low_confidence=False,
    )
    return _interaction_result_to_text_http_response(
        interaction_service.build_response(core_decision)
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile | None = File(default=None),
    x_rocky_device: str = Header(default="rocky-01"),
    x_rocky_interaction_mode: InteractionMode | None = Header(default=None),
    x_rocky_device_state: str = Header(default="active"),
) -> dict[str, object]:
    received_at = conversation_manager.capture_request_time()
    with latency_scope():
        return await _transcribe(
            file,
            x_rocky_device,
            received_at,
            x_rocky_interaction_mode,
            _is_sleeping_device_state(x_rocky_device_state),
        )


async def _transcribe(
    file: UploadFile | None, x_rocky_device: str, received_at: float,
    interaction_mode: InteractionMode | None = None, device_sleeping: bool = False,
) -> dict[str, object]:
    audio_data = await read_wav_upload(file)
    duration_seconds = validate_wav(audio_data)
    technical("[STT] request received")
    technical(f"[STT] wav: {SAMPLE_RATE} Hz mono 16-bit duration={duration_seconds:.1f}s")

    use_wake_hotword = _use_wake_hotword(x_rocky_device, interaction_mode)
    technical(f"[STT] mode={'wake' if use_wake_hotword else 'command'}")
    mark_latency("stt_start")
    transcription = transcribe_audio(audio_data, file.filename, use_wake_hotword)
    text, low_confidence = _conversation_input(transcription, use_wake_hotword)
    mark_latency("stt_done")
    latency_metric("stt_ms", "stt_start", "stt_done")
    technical("[STT] transcription completed")
    core_decision = _classify_conversation(
        text=text,
        device_id=x_rocky_device,
        received_at=received_at,
        interaction_mode=interaction_mode,
        low_confidence=low_confidence,
        device_sleeping=device_sleeping,
    )
    result = interaction_service.build_response(core_decision)
    response = _interaction_result_to_http_response(result)
    _report_transcription_and_response(text, low_confidence, response)
    response["audio_available"] = response_speaker.speak_response(response, text)
    report_latency_summary()
    return response


@app.post("/transcribe/realtime")
async def transcribe_realtime(
    file: UploadFile | None = File(default=None),
    x_rocky_device: str = Header(default="rocky-01"),
    x_rocky_interaction_mode: InteractionMode | None = Header(default=None),
    x_rocky_device_state: str = Header(default="active"),
) -> dict[str, object]:
    received_at = conversation_manager.capture_request_time()
    with latency_scope():
        return await _transcribe_realtime(
            file,
            x_rocky_device,
            received_at,
            x_rocky_interaction_mode,
            _is_sleeping_device_state(x_rocky_device_state),
        )


async def _transcribe_realtime(
    file: UploadFile | None, x_rocky_device: str, received_at: float,
    interaction_mode: InteractionMode | None = None, device_sleeping: bool = False,
) -> dict[str, object]:
    audio_data = await read_wav_upload(file)
    duration_seconds = validate_wav(audio_data)
    technical("[RTD] request received")
    technical(f"[RTD] wav duration={duration_seconds:.1f}s")
    use_wake_hotword = _use_wake_hotword(x_rocky_device, interaction_mode)
    mark_latency("stt_start")
    transcription = transcribe_audio(audio_data, file.filename, use_wake_hotword)
    text, low_confidence = _conversation_input(transcription, use_wake_hotword)
    mark_latency("stt_done")
    latency_metric("stt_ms", "stt_start", "stt_done")
    core_decision = _classify_conversation(
        text=text,
        device_id=x_rocky_device,
        received_at=received_at,
        interaction_mode=interaction_mode,
        low_confidence=low_confidence,
        device_sleeping=device_sleeping,
    )
    technical(f"[RTD] classify kind={core_decision.kind}")
    if core_decision.direct_result is not None:
        result = interaction_service.build_direct_response(core_decision)
        response = _interaction_result_to_http_response(result)
        _report_transcription_and_response(
            text, low_confidence, response, InteractionKind(core_decision.kind)
        )
        interaction_id = realtime_audio_service.queue_direct_response(response, text)
        report_latency_summary()
        return {
            "realtime": interaction_id is not None,
            "interaction_id": interaction_id,
            **response,
            "audio_available": False,
        }

    interaction_id = realtime_audio_service.create_interaction()
    info(f'Ouvi: "{text}"')
    technical(f"[RTD] interaction_created id={interaction_id}")
    report_latency("interaction_id", interaction_id)

    def produce() -> None:
        try:
            realtime_audio_service.build_stream_audio_response_from_core_decision(
                interaction_service, core_decision, interaction_id,
            )
        except Exception as exception:
            log_error("Falha ao gerar áudio realtime.")
            technical(f"[RTD] producer_error type={type(exception).__name__}")
        finally:
            realtime_audio_service.audio_queue.finish_if_active(interaction_id, failed=True)

    request_context = copy_context()
    thread_started = False
    try:
        Thread(target=lambda: request_context.run(produce), name=f"rocky-realtime-{interaction_id}", daemon=True).start()
        thread_started = True
    except Exception as exception:
        log_error("Falha ao iniciar áudio realtime.")
        technical(f"[RTD] thread_start_error type={type(exception).__name__}")
    finally:
        if not thread_started:
            realtime_audio_service.audio_queue.finish_if_active(interaction_id, failed=True)
    return {
        "realtime": True,
        "interaction_id": interaction_id,
        "text": "",
        "interaction_state": "thinking",
        "actions": [],
        "audio_available": False,
    }


def _use_wake_hotword(device_id: str, interaction_mode: InteractionMode | None) -> bool:
    if interaction_mode is not None:
        return interaction_mode == InteractionMode.WAKE
    return not command_session.is_active(device_id)


def _classify_conversation(
    *,
    text: str,
    device_id: str,
    received_at: float,
    interaction_mode: InteractionMode | None,
    low_confidence: bool,
    device_sleeping: bool = False,
):
    """Conecta a sessão de dispositivo ao Core sem levar estado ao Core."""
    if interaction_mode == InteractionMode.WAKE:
        command_session.close(device_id)

    effective_mode = interaction_mode
    if effective_mode is None and command_session.is_active(device_id):
        effective_mode = InteractionMode.COMMAND

    if device_sleeping:
        sleeping_decision = _sleeping_device_decision(text, device_id, effective_mode)
        if sleeping_decision is not None:
            command_session.close(device_id)
            return sleeping_decision

    core_decision = interaction_service.classify(
        user_id=LOCAL_USER_ID,
        device_id=device_id,
        input_type="voice_transcript",
        text=text,
        received_at=received_at,
        interaction_mode=effective_mode,
        low_confidence=low_confidence,
    )

    if core_decision.kind == InteractionKind.WAKE.value:
        command_session.open(device_id)
    elif effective_mode == InteractionMode.COMMAND:
        direct_result = core_decision.direct_result
        if direct_result is None or direct_result.interaction_state != "attention":
            command_session.close(device_id)

    return core_decision


def _is_sleeping_device_state(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized == "sleeping":
        return True
    if normalized == "active":
        return False
    raise HTTPException(status_code=422, detail="X-Rocky-Device-State invalido.")


def _sleeping_device_decision(
    text: str, device_id: str, interaction_mode: InteractionMode | None,
) -> CoreDecision | None:
    """Barreira de dispositivo: evita Core/LLM para fala comum durante o sono."""
    normalized = normalize_text(text)
    wake_command = extract_wake_command(text)
    if wake_command == "acorda" or (
        interaction_mode is InteractionMode.COMMAND and normalized == "acorda"
    ):
        return _sleeping_direct_decision(
            "deterministic",
            text,
            device_id,
            "Acordando.",
            [{"type": "set_device_state", "state": "active"}],
        )

    if interaction_mode is InteractionMode.WAKE and is_wake_phrase(text):
        return None

    return _sleeping_direct_decision("empty", text, device_id, "", [])


def _sleeping_direct_decision(
    kind: str, text: str, device_id: str, response_text: str, actions: list[dict[str, object]],
) -> CoreDecision:
    return CoreDecision(
        kind=kind,
        text=text,
        device_id=device_id,
        user_id=LOCAL_USER_ID,
        direct_result=InteractionResult(
            text=response_text,
            interaction_state="idle",
            actions=actions,
        ),
    )


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
            technical(f"[RTD] served id={interaction_id} seq={item.sequence}")
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
            technical(f"[RTD] done id={interaction_id}")
            return _realtime_status("done")
        if state.production_status is AudioProductionStatus.FAILED:
            return _realtime_status("failed")
        if state.production_status is AudioProductionStatus.CANCELLED:
            return _realtime_status("cancelled")
        if (time.monotonic() - started_at) * 1000 >= timeout_ms:
            technical(f"[RTD] next_wait_ms={round((time.monotonic() - started_at) * 1000)}")
            return _realtime_status("pending")
        time.sleep(REALTIME_LONG_POLL_INTERVAL_MS / 1000)


@app.post("/realtime/audio/{interaction_id}/{sequence}/consumed")
def acknowledge_realtime_audio(interaction_id: int, sequence: int) -> dict[str, int | str]:
    try:
        realtime_audio_service.audio_queue.mark_consumed(interaction_id, sequence)
    except ValueError as error:
        raise HTTPException(status_code=409, detail="ACK realtime invalido.") from error
    technical(f"[RTD] consumed id={interaction_id} seq={sequence}")
    return {"status": "consumed", "interaction_id": interaction_id, "sequence": sequence}


def _realtime_status(status: str) -> JSONResponse:
    return JSONResponse({"status": status})


@app.get("/audio/rocky_response.wav")
def get_response_audio() -> FileResponse:
    if not TTS_OUTPUT_PATH.is_file() or TTS_OUTPUT_PATH.stat().st_size == 0:
        raise HTTPException(status_code=404, detail="Audio indisponivel.")

    technical(f"[AUDIO] served path={TTS_OUTPUT_PATH.name} bytes={TTS_OUTPUT_PATH.stat().st_size}")
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
    technical(f"[DATASET] label={label}")
    technical(f"[DATASET] saved={destination.name}")
    technical(f"[DATASET] duration={duration_seconds:.2f}s")
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
    technical(f"[CAL] saved={destination.name}")
    technical(f"[CAL] pcm16_rms={pcm_metrics['rms']} peak_abs={pcm_metrics['peak_abs']}")
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
