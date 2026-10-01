"""Orquestra a síntese de respostas do Rocky e o playback local opcional."""

import audioop
import os
import struct
import winsound
import wave
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from models.actions import BackendResponse
from services.tts.base import TTSClient


SUPPORTED_SAMPLE_RATE = 32000
FULL_SCALE = 32767


@dataclass(frozen=True)
class PcmMetrics:
    minimum: int
    maximum: int
    peak_absolute: int
    mean: float
    rms: float
    negative_clipping: int
    positive_clipping: int
    sample_count: int
    above_90_percent: int
    above_95_percent: int
    above_99_percent: int

    @property
    def clipping_percent(self) -> float:
        return (self.negative_clipping + self.positive_clipping) * 100.0 / self.sample_count

    def percentage(self, count: int) -> float:
        return count * 100.0 / self.sample_count


def get_spoken_text(response: BackendResponse, request_text: str) -> str | None:
    """Seleciona apenas o conteúdo destinado ao usuário para reprodução local."""
    for action in response.get("actions", []):
        if action["type"] == "show_text":
            return " ".join((action["line1"], action["line2"])).strip() or None

    response_text = response.get("text", "").strip()
    if not response_text or response_text == request_text.strip():
        return None

    return response_text


class LocalResponseSpeaker:
    def __init__(
        self,
        tts_client: TTSClient,
        output_path: Path,
        enable_local_playback: bool = True,
    ) -> None:
        self._tts_client = tts_client
        self._output_path = output_path
        self._enable_local_playback = enable_local_playback
        self._lock = Lock()

    def speak_response(self, response: BackendResponse, request_text: str) -> bool:
        spoken_text = get_spoken_text(response, request_text)
        if spoken_text is None:
            return False

        try:
            with self._lock:
                self._tts_client.synthesize_to_wav(spoken_text, self._output_path)
                resample_wav(self._output_path, SUPPORTED_SAMPLE_RATE)
                print(f"[TTS] generated path={self._output_path.name} rate={SUPPORTED_SAMPLE_RATE}")
                report_pcm_metrics(analyze_pcm_wav(self._output_path))

                if self._enable_local_playback:
                    winsound.PlaySound(str(self._output_path), winsound.SND_FILENAME)
        except Exception as error:
            print(f"[TTS] error={error}")
            return False

        print(f'[AUDIO] ready path={self._output_path.name}')
        return True


def get_local_playback_enabled() -> bool:
    value = os.getenv("ROCKY_LOCAL_TTS_PLAYBACK", "1").strip().casefold()
    return value not in {"0", "false", "no", "off"}


def resample_wav(wav_path: Path, target_sample_rate: int) -> None:
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_sample_rate = wav_file.getframerate()
        compression = wav_file.getcomptype()
        pcm_data = wav_file.readframes(wav_file.getnframes())

    if compression != "NONE" or sample_width != 2 or channels != 1:
        raise ValueError("O WAV do Piper deve ser PCM mono de 16 bits.")
    if source_sample_rate == target_sample_rate:
        return

    resampled_data, _ = audioop.ratecv(
        pcm_data,
        sample_width,
        channels,
        source_sample_rate,
        target_sample_rate,
        None,
    )
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(target_sample_rate)
        wav_file.writeframes(resampled_data)


def analyze_pcm_wav(wav_path: Path) -> PcmMetrics:
    with wave.open(str(wav_path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE" or wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise ValueError("O WAV analisado deve ser PCM mono de 16 bits.")
        pcm_data = wav_file.readframes(wav_file.getnframes())

    sample_count = len(pcm_data) // 2
    if sample_count == 0:
        raise ValueError("O WAV analisado não contém amostras.")

    minimum = FULL_SCALE
    maximum = -FULL_SCALE - 1
    sample_sum = 0
    sample_squares = 0
    negative_clipping = 0
    positive_clipping = 0
    above_90_percent = 0
    above_95_percent = 0
    above_99_percent = 0
    threshold_90 = FULL_SCALE * 0.90
    threshold_95 = FULL_SCALE * 0.95
    threshold_99 = FULL_SCALE * 0.99

    for (sample,) in struct.iter_unpack("<h", pcm_data):
        minimum = min(minimum, sample)
        maximum = max(maximum, sample)
        sample_sum += sample
        sample_squares += sample * sample
        absolute_sample = abs(sample)
        negative_clipping += sample == -32768
        positive_clipping += sample == 32767
        above_90_percent += absolute_sample >= threshold_90
        above_95_percent += absolute_sample >= threshold_95
        above_99_percent += absolute_sample >= threshold_99

    return PcmMetrics(
        minimum=minimum,
        maximum=maximum,
        peak_absolute=max(abs(minimum), abs(maximum)),
        mean=sample_sum / sample_count,
        rms=(sample_squares / sample_count) ** 0.5,
        negative_clipping=negative_clipping,
        positive_clipping=positive_clipping,
        sample_count=sample_count,
        above_90_percent=above_90_percent,
        above_95_percent=above_95_percent,
        above_99_percent=above_99_percent,
    )


def report_pcm_metrics(metrics: PcmMetrics) -> None:
    print(
        "[TTS PCM] "
        f"min={metrics.minimum} max={metrics.maximum} peak={metrics.peak_absolute} "
        f"mean={metrics.mean:.1f} rms={metrics.rms:.1f} "
        f"clipping={metrics.negative_clipping + metrics.positive_clipping} "
        f"({metrics.clipping_percent:.4f}%)"
    )
    print(
        "[TTS PCM] "
        f"above90={metrics.percentage(metrics.above_90_percent):.4f}% "
        f"above95={metrics.percentage(metrics.above_95_percent):.4f}% "
        f"above99={metrics.percentage(metrics.above_99_percent):.4f}%"
    )
