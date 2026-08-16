"""Contrato independente para mecanismos de síntese de fala."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TTSClient(Protocol):
    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        """Sintetiza texto em um arquivo WAV local."""
