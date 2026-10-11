"""Saída humana do backend e diagnósticos técnicos opcionais."""

import os


_VERBOSE_VALUES = frozenset({"1", "true", "yes", "on"})


def is_verbose(value: str | None = None) -> bool:
    """Interpreta KODA_VERBOSE sem capturar seu valor durante o import."""
    candidate = os.getenv("KODA_VERBOSE", "") if value is None else value
    return candidate.strip().casefold() in _VERBOSE_VALUES


def technical(message: str) -> None:
    """Emite diagnóstico somente quando KODA_VERBOSE está ativo."""
    if is_verbose():
        print(message)


def info(message: str) -> None:
    """Emite uma mensagem curta destinada ao operador do Koda."""
    print(f"[KODA] {message}")


def error(message: str) -> None:
    """Erros importantes não dependem do modo verbose."""
    print(f"[KODA] {message}")
