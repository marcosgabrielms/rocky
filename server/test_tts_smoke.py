"""Gera e valida um WAV local do Piper para o Rocky."""

import wave
from pathlib import Path

from services.tts import PiperTTS


TEXT = "Olá. Eu sou o Rocky."
TTS_DIRECTORY = Path(__file__).parent / "local" / "tts"
MODEL_PATH = TTS_DIRECTORY / "voices" / "pt_BR-jeff-medium.onnx"
OUTPUT_PATH = TTS_DIRECTORY / "rocky_piper_test.wav"


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelo Piper não encontrado: {MODEL_PATH}")

    PiperTTS(MODEL_PATH).synthesize_to_wav(TEXT, OUTPUT_PATH)
    validate_wav(OUTPUT_PATH)


def validate_wav(wav_path: Path) -> None:
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise ValueError("O WAV de teste não foi gerado corretamente.")

    with wave.open(str(wav_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        duration_seconds = frames / sample_rate if sample_rate else 0.0

    if frames == 0 or sample_rate == 0:
        raise ValueError("O WAV de teste não contém áudio válido.")

    print(f"wav={wav_path}")
    print(f"bytes={wav_path.stat().st_size}")
    print(f"sample_rate={sample_rate}")
    print(f"duration_seconds={duration_seconds:.2f}")


if __name__ == "__main__":
    main()
