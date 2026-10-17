"""Comandos determinísticos executados localmente pelo backend."""

import re
import unicodedata
from datetime import datetime
from enum import StrEnum
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


class DeviceStateIntent(StrEnum):
    WAKE_DEVICE = "wake_device"
    SLEEP_DEVICE = "sleep_device"


WAKE_DEVICE_KEYWORDS = frozenset({"acorda", "acordar", "acorde", "desperta", "despertar"})
SLEEP_DEVICE_KEYWORDS = frozenset({"dorme", "dormir", "durma", "descansa", "descansar"})
DEVICE_STATE_HELPERS = frozenset({"koda", "voce", "pode", "ai", "por", "favor", "vai", "va"})
SHORT_COMMAND_ALLOWLIST = frozenset({"koda", "sim", "nao", "pode", "pare", "azul", "hora", "horas"})


def normalize_text(text: str) -> str:
    return text.strip().casefold().rstrip(".?!,;:")


def detect_command_intent(normalized_text: str) -> str | None:
    device_state_intent = detect_device_state_intent(normalized_text)
    if device_state_intent is DeviceStateIntent.SLEEP_DEVICE:
        return "sleep"
    if device_state_intent is DeviceStateIntent.WAKE_DEVICE:
        return "wake"
    if normalized_text in HOURS_ALIASES:
        return "hours"
    if extract_volume_percent(normalized_text) is not None:
        return "volume"
    return None


def normalize_device_state_text(text: str) -> str:
    """Normaliza somente comandos de estado, sem mudar os demais parsers."""
    decomposed = unicodedata.normalize("NFD", text).casefold()
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", without_accents).split())


def detect_device_state_intent(
    text: str,
    *,
    require_wake_word: bool = False,
    allow_stt_recovery: bool = False,
    device_sleeping: bool = False,
) -> DeviceStateIntent | None:
    """Reconhece comandos físicos; recuperação exige contexto de janela válido."""
    tokens = normalize_device_state_text(text).split()
    if not tokens or (require_wake_word and "koda" not in tokens):
        return None

    wake_keywords = [token for token in tokens if _matches_state_keyword(token, WAKE_DEVICE_KEYWORDS)]
    sleep_keywords = [token for token in tokens if _matches_state_keyword(token, SLEEP_DEVICE_KEYWORDS)]
    if len(wake_keywords) + len(sleep_keywords) == 1:
        for token in tokens:
            if token in DEVICE_STATE_HELPERS:
                continue
            if _matches_state_keyword(token, WAKE_DEVICE_KEYWORDS | SLEEP_DEVICE_KEYWORDS):
                continue
            break
        else:
            return DeviceStateIntent.WAKE_DEVICE if wake_keywords else DeviceStateIntent.SLEEP_DEVICE

    if not allow_stt_recovery or require_wake_word:
        return None

    if device_sleeping:
        return DeviceStateIntent.WAKE_DEVICE if _matches_wake_stt_recovery(tokens) else None
    return DeviceStateIntent.SLEEP_DEVICE if _matches_sleep_stt_recovery(tokens) else None


def is_ambiguous_short_command_noise(text: str) -> bool:
    """Evita enviar ao modelo ruído muito curto logo após uma wake válida."""
    tokens = normalize_device_state_text(text).split()
    if not tokens or detect_command_intent(" ".join(tokens)) is not None:
        return False
    if len(tokens) == 1:
        return 2 <= len(tokens[0]) <= 4 and tokens[0] not in SHORT_COMMAND_ALLOWLIST
    return len(tokens) <= 3 and tokens[0] in {"vai", "va", "hora"}


def _matches_sleep_stt_recovery(tokens: list[str]) -> bool:
    """Recupera apenas o padrão curto observado 'vai do ...' em janela ativa."""
    if not 2 <= len(tokens) <= 3 or tokens[0] not in {"vai", "va"}:
        return False
    compact_tail = "".join(tokens[1:])
    return compact_tail in {"do", "dob", "dorigido"}


def _matches_wake_stt_recovery(tokens: list[str]) -> bool:
    """Aceita erro fonético curto de 'acorda' somente enquanto o dispositivo dorme."""
    if len(tokens) != 1:
        return False
    consonants = "".join(character for character in tokens[0] if character not in "aeiou")
    return bool(consonants) and _bounded_edit_distance(consonants, "crd", maximum=1) <= 1


def _matches_state_keyword(token: str, keywords: frozenset[str]) -> bool:
    if token in keywords:
        return True
    if len(token) < 5:
        return False
    return any(_bounded_edit_distance(token, keyword, maximum=1) <= 1 for keyword in keywords)


def _bounded_edit_distance(left: str, right: str, *, maximum: int) -> int:
    if abs(len(left) - len(right)) > maximum:
        return maximum + 1

    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        row_minimum = current[0]
        for right_index, right_character in enumerate(right, start=1):
            cost = 0 if left_character == right_character else 1
            value = min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + cost,
            )
            current.append(value)
            row_minimum = min(row_minimum, value)
        if row_minimum > maximum:
            return maximum + 1
        previous = current
    return previous[-1]


def extract_volume_percent(normalized_text: str) -> int | None:
    match = VOLUME_COMMAND_PATTERN.fullmatch(normalized_text)
    return int(match.group("percent")) if match is not None else None


def get_current_time() -> str:
    return datetime.now(ZoneInfo(SAO_PAULO_TIMEZONE_NAME)).strftime("%H:%M")
