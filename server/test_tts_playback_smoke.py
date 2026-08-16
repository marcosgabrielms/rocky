"""Smoke test manual da reprodução local do Piper no Windows."""

from pathlib import Path

from services.tts import LocalResponseSpeaker, PiperTTS


TEXT = "Olá. Esta é uma resposta do Rocky."
TTS_DIRECTORY = Path(__file__).parent / "local" / "tts"
MODEL_PATH = TTS_DIRECTORY / "voices" / "pt_BR-jeff-medium.onnx"
OUTPUT_PATH = TTS_DIRECTORY / "rocky_response.wav"


def main() -> None:
    speaker = LocalResponseSpeaker(PiperTTS(MODEL_PATH), OUTPUT_PATH)
    response = {"text": TEXT, "interaction_state": "idle", "actions": []}
    if not speaker.speak_response(response, "Pergunta de teste"):
        raise RuntimeError("A reprodução local do Rocky falhou.")

    print(f"wav={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
