"""Medições monotônicas locais, isoladas por requisição e sem mudar o protocolo."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count

from services.runtime_log import info, technical

_request_ids = count(1)


class LatencyTrace:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._events: dict[str, int] = {}
        self.request_id = next(_request_ids)

    def mark(self, event: str) -> bool:
        if event in self._events:
            return False
        elapsed_ms = round((self._clock() - self._started_at) * 1000)
        self._events[event] = elapsed_ms
        self.report(f"{event}_ms", elapsed_ms, kind="event")
        return True

    def metric(self, name: str, start: str, end: str) -> None:
        if start in self._events and end in self._events:
            self.report(name, self._events[end] - self._events[start], kind="duration")

    def report(self, name: str, value: object, kind: str = "info") -> None:
        technical(f"[KODA LATENCY] clock=backend_request request={self.request_id} kind={kind} {name}={value}")

    def summary(self) -> None:
        """Mostra uma linha curta com as durações já medidas nesta interação."""
        fields = (("STT", "stt_ms"), ("GPT", "llm_first_token_ms"), ("TTS", "tts_first_sentence_ms"))
        parts = [f"{label} {self._events[name] / 1000:.1f}s" for label, name in fields if name in self._events]
        if parts:
            info("Latência: " + " | ".join(parts))


_active_trace: ContextVar[LatencyTrace | None] = ContextVar("koda_latency", default=None)


@contextmanager
def latency_scope(clock: Callable[[], float] = time.monotonic) -> Iterator[LatencyTrace]:
    trace = LatencyTrace(clock)
    token = _active_trace.set(trace)
    try:
        trace.mark("request_received")
        yield trace
    finally:
        _active_trace.reset(token)


def mark_latency(event: str) -> bool:
    trace = _active_trace.get()
    return trace.mark(event) if trace is not None else False


def latency_metric(name: str, start: str, end: str) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.metric(name, start, end)


def report_latency(name: str, value: object) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.report(name, value)


def report_latency_summary() -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.summary()
