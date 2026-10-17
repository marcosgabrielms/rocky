"""Comandos determinísticos executados localmente pelo backend."""

import re
from datetime import datetime
from zoneinfo import ZoneInfo


HOURS_ALIASES = frozenset({
    "horas", "hora", "hores", "oras", "ors",
    "que horas são", "que horas sao", "que horas é", "que horas e",
    "qual a hora", "qual é a hora", "qual e a hora",
    "me diga as horas", "me diga a hora",
})
SAO_PAULO_TIMEZONE_NAME = "America/Sao_Paulo"
VOLUME_COMMAND_PATTERN = re.compile(
    r"^(?:(?:(?:coloque|coloca)\s+)?(?:o\s+)?volume|(?:aumenta|abaixa)\s+(?:o\s+)?volume)"
    r"\s+(?:em|para)\s+(?P<percent>\d+)\s*%$"
)
SLEEP_ALIASES = frozenset({"vai dormir", "va dormir", "dorme"})


def normalize_text(text: str) -> str:
    return text.strip().casefold().rstrip(".?!,;:")


def detect_command_intent(normalized_text: str) -> str | None:
    if normalized_text in SLEEP_ALIASES:
        return "sleep"
    if normalized_text in HOURS_ALIASES:
        return "hours"
    if extract_volume_percent(normalized_text) is not None:
        return "volume"
    return None


def extract_volume_percent(normalized_text: str) -> int | None:
    match = VOLUME_COMMAND_PATTERN.fullmatch(normalized_text)
    return int(match.group("percent")) if match is not None else None


def get_current_time() -> str:
    return datetime.now(ZoneInfo(SAO_PAULO_TIMEZONE_NAME)).strftime("%H:%M")
