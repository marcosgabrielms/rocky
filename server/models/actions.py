"""Contrato de responses e actions aceito pelo ESP32."""

from typing import TypedDict


class ExpressionAction(TypedDict):
    type: str
    value: str


class ShowTextAction(TypedDict):
    type: str
    line1: str
    line2: str
    duration_ms: int


class SetVolumeAction(TypedDict):
    type: str
    percent: int


class BackendResponse(TypedDict, total=False):
    text: str
    interaction_state: str
    command_window_ms: int
    actions: list[ExpressionAction | ShowTextAction | SetVolumeAction]
    audio_available: bool


def expression(value: str) -> ExpressionAction:
    return {"type": "expression", "value": value}


def show_text(line1: str, line2: str, duration_ms: int) -> ShowTextAction:
    return {
        "type": "show_text",
        "line1": line1,
        "line2": line2,
        "duration_ms": duration_ms,
    }


def set_volume(percent: int) -> SetVolumeAction:
    return {"type": "set_volume", "percent": percent}


def idle_response(text: str) -> BackendResponse:
    return {
        "text": text,
        "interaction_state": "idle",
        "actions": [expression("idle")],
    }
