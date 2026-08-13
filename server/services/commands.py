"""Comandos determinísticos executados localmente pelo backend."""

from datetime import datetime
from zoneinfo import ZoneInfo


HOURS_ALIASES = frozenset({"horas", "hora", "hores", "oras", "ors"})
SAO_PAULO_TIMEZONE_NAME = "America/Sao_Paulo"


def normalize_text(text: str) -> str:
    return text.strip().casefold().rstrip(".?!,;:")


def detect_command_intent(normalized_text: str) -> str | None:
    if normalized_text in HOURS_ALIASES:
        return "hours"
    return None


def get_current_time() -> str:
    return datetime.now(ZoneInfo(SAO_PAULO_TIMEZONE_NAME)).strftime("%H:%M")
