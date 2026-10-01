"""Serviços de síntese de fala do Rocky."""

from services.tts.base import TTSClient
from services.tts.piper import PiperTTS
from services.tts.response_speaker import LocalResponseSpeaker, get_local_playback_enabled, get_spoken_text

__all__ = ["LocalResponseSpeaker", "PiperTTS", "TTSClient", "get_local_playback_enabled", "get_spoken_text"]
