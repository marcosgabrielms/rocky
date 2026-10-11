"""Gera candidatos locais de voz para avaliação offline do Koda."""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path


PHRASE = "Olá, eu sou o Kôda. Estou ouvindo você. O que vamos fazer hoje?"
SERVER_DIRECTORY = Path(__file__).resolve().parent.parent
if str(SERVER_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SERVER_DIRECTORY))

MODEL_PATH = SERVER_DIRECTORY / "local" / "tts" / "voices" / "pt_BR-jeff-medium.onnx"
OUTPUT_DIRECTORY = SERVER_DIRECTORY / "voice_lab" / "output"
TARGET_LOUDNESS = "I=-18:TP=-2:LRA=7"
NEGLIGIBLE_CLIPPING_SAMPLES = 1


@dataclass(frozen=True)
class VoiceCandidate:
    filename: str
    description: str
    filters: str | None


CANDIDATES = (
    VoiceCandidate("koda_baseline.wav", "Piper puro, sem pós-processamento.", None),
    VoiceCandidate(
        "koda_natural.wav",
        "EQ de clareza e compressão muito suave.",
        "highpass=f=70,equalizer=f=2800:t=q:w=1.2:g=1.0,"
        "acompressor=threshold=-18dB:ratio=1.25:attack=20:release=160:makeup=1.3,"
        f"loudnorm={TARGET_LOUDNESS}",
    ),
    VoiceCandidate(
        "koda_balanced.wav",
        "Pitch +0,75 semitom, EQ leve, compressão suave e assinatura discreta.",
        "asetrate=22050*1.04427,aresample=22050,atempo=0.95761,"
        "highpass=f=70,equalizer=f=3000:t=q:w=1.2:g=1.2,"
        "acompressor=threshold=-19dB:ratio=1.35:attack=18:release=150:makeup=1.4,"
        f"loudnorm={TARGET_LOUDNESS}",
    ),
    VoiceCandidate(
        "koda_synthetic.wav",
        "Pitch +1,25 semitom, clareza e doubling eletrônico leve.",
        "asetrate=22050*1.07430,aresample=22050,atempo=0.93084,"
        "highpass=f=80,equalizer=f=3100:t=q:w=1.1:g=1.4,"
        "acompressor=threshold=-20dB:ratio=1.45:attack=16:release=140:makeup=1.5,"
        "chorus=0.25:0.35:12:0.12:0.18:2,"
        f"loudnorm={TARGET_LOUDNESS}",
    ),
)


@dataclass(frozen=True)
class WavMetrics:
    path: Path
    bytes: int
    sample_rate: int
    channels: int
    duration_seconds: float
    minimum: int
    maximum: int
    sample_count: int
    clipping_samples: int
    clipping_percent: float
    clipping_status: str


def main() -> None:
    argparse.ArgumentParser(
        description="Gera candidatos locais de voz do Koda para avaliação offline."
    ).parse_args()
    require_prerequisites()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    baseline_path = OUTPUT_DIRECTORY / CANDIDATES[0].filename
    synthesize_baseline(baseline_path)

    metrics = [validate_wav(baseline_path)]
    for candidate in CANDIDATES[1:]:
        output_path = OUTPUT_DIRECTORY / candidate.filename
        apply_ffmpeg_filters(baseline_path, output_path, candidate.filters)
        metrics.append(validate_wav(output_path))

    write_manifest(metrics)
    for metric in metrics:
        print(
            f"wav={metric.path.name} rate={metric.sample_rate}Hz channels={metric.channels} "
            f"duration={metric.duration_seconds:.2f}s bytes={metric.bytes} "
            f"min={metric.minimum} max={metric.maximum} samples={metric.sample_count} "
            f"clipping={metric.clipping_samples} ({metric.clipping_percent:.6f}%) "
            f"status={metric.clipping_status}"
        )


def require_prerequisites() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelo Piper não encontrado: {MODEL_PATH}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg não está disponível. O laboratório não instala ferramentas automaticamente."
        )


def synthesize_baseline(output_path: Path) -> None:
    from services.tts.piper import PiperTTS

    PiperTTS(MODEL_PATH).synthesize_to_wav(PHRASE, output_path)


def apply_ffmpeg_filters(source_path: Path, output_path: Path, filters: str | None) -> None:
    if filters is None:
        raise ValueError("O baseline não usa filtros FFmpeg.")

    subprocess.run(
        (
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-af",
            filters,
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output_path),
        ),
        check=True,
    )


def validate_wav(wav_path: Path) -> WavMetrics:
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise ValueError(f"WAV ausente ou vazio: {wav_path}")

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        compression = wav_file.getcomptype()
        pcm_data = wav_file.readframes(frames)

    if compression != "NONE" or channels != 1 or sample_width != 2 or sample_rate <= 0 or frames <= 0:
        raise ValueError(f"WAV inválido: {wav_path.name}")

    samples = tuple(sample[0] for sample in struct.iter_unpack("<h", pcm_data))
    minimum = min(samples)
    maximum = max(samples)
    sample_count = len(samples)
    clipping_samples = sum(sample in (-32768, 32767) for sample in samples)
    clipping_percent = clipping_samples * 100 / sample_count
    clipping_status = classify_clipping(clipping_samples)
    if clipping_status == "ERROR":
        raise ValueError(
            f"Clipping relevante em {wav_path.name}: {clipping_samples}/{sample_count} "
            f"amostras ({clipping_percent:.6f}%)."
        )
    if clipping_status == "WARNING":
        print(
            f"WARNING: pico isolado em {wav_path.name}: {clipping_samples}/{sample_count} "
            f"amostras ({clipping_percent:.6f}%)."
        )

    return WavMetrics(
        path=wav_path,
        bytes=wav_path.stat().st_size,
        sample_rate=sample_rate,
        channels=channels,
        duration_seconds=frames / sample_rate,
        minimum=minimum,
        maximum=maximum,
        sample_count=sample_count,
        clipping_samples=clipping_samples,
        clipping_percent=clipping_percent,
        clipping_status=clipping_status,
    )


def classify_clipping(clipping_samples: int) -> str:
    if clipping_samples == 0:
        return "OK"
    if clipping_samples <= NEGLIGIBLE_CLIPPING_SAMPLES:
        return "WARNING"
    return "ERROR"


def write_manifest(metrics: list[WavMetrics]) -> None:
    candidates_by_name = {candidate.filename: candidate for candidate in CANDIDATES}
    lines = [
        "# Koda Voice Lab",
        "",
        f"Frase: {PHRASE}",
        f"Voz-base: {MODEL_PATH.name}",
        "",
    ]
    for metric in metrics:
        candidate = candidates_by_name[metric.path.name]
        lines.extend(
            (
                f"## {metric.path.name}",
                f"- perfil: {candidate.description}",
                f"- sample rate: {metric.sample_rate} Hz",
                f"- canais: {metric.channels}",
                f"- duração: {metric.duration_seconds:.2f} s",
                f"- amostras: {metric.sample_count}",
                f"- clipping: {metric.clipping_samples} ({metric.clipping_percent:.6f}%)",
                f"- status de clipping: {metric.clipping_status}",
                "",
            )
        )
    (OUTPUT_DIRECTORY / "manifest.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
