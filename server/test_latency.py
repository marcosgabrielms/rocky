"""Testes de relógio, isolamento e propagação do diagnóstico sem serviços reais."""

import io
import os
import unittest
from contextlib import redirect_stdout
from contextvars import copy_context
from unittest.mock import patch

from services.latency import LatencyTrace, latency_metric, latency_scope, mark_latency


class LatencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.output = io.StringIO()
        self._verbose = patch.dict(os.environ, {"KODA_VERBOSE": "1"})
        self._verbose.start()

    def tearDown(self) -> None:
        self._verbose.stop()

    def clock(self) -> float:
        return self.now

    def test_monotonic_events_and_stage_duration_use_distinct_kinds(self) -> None:
        with redirect_stdout(self.output):
            trace = LatencyTrace(self.clock)
            trace.mark("stt_start")
            self.now += 0.125
            trace.mark("stt_done")
            trace.metric("stt_ms", "stt_start", "stt_done")
        log = self.output.getvalue()
        self.assertIn("kind=event stt_done_ms=125", log)
        self.assertIn("kind=duration stt_ms=125", log)

    def test_first_event_is_not_overwritten_by_later_segments(self) -> None:
        with redirect_stdout(self.output), latency_scope(self.clock):
            self.assertTrue(mark_latency("tts_start"))
            self.now += 10
            self.assertFalse(mark_latency("tts_start"))
        self.assertEqual(self.output.getvalue().count("tts_start_ms="), 1)

    def test_missing_stage_does_not_produce_a_fabricated_metric(self) -> None:
        with redirect_stdout(self.output), latency_scope(self.clock):
            latency_metric("stt_ms", "stt_start", "stt_done")
        self.assertNotIn("stt_ms=", self.output.getvalue())

    def test_scopes_restore_previous_request_and_are_isolated(self) -> None:
        with redirect_stdout(self.output), latency_scope(self.clock) as outer:
            mark_latency("stt_start")
            with latency_scope(self.clock) as inner:
                self.assertNotEqual(outer.request_id, inner.request_id)
                mark_latency("stt_done")
                latency_metric("stt_ms", "stt_start", "stt_done")
            self.now += 0.050
            mark_latency("stt_done")
            latency_metric("stt_ms", "stt_start", "stt_done")
        self.assertEqual(self.output.getvalue().count("kind=duration stt_ms="), 1)
        self.assertIn("kind=duration stt_ms=50", self.output.getvalue())

    def test_copied_context_keeps_trace_after_http_scope_exits(self) -> None:
        with redirect_stdout(self.output):
            with latency_scope(self.clock) as trace:
                mark_latency("llm_request_start")
                context = copy_context()
            self.now += 0.200
            context.run(mark_latency, "llm_first_token")
            context.run(latency_metric, "llm_first_token_ms", "llm_request_start", "llm_first_token")
            self.assertFalse(mark_latency("outside_request"))
        self.assertIn(f"request={trace.request_id} kind=duration llm_first_token_ms=200", self.output.getvalue())

    def test_diagnostic_can_be_disabled_without_changing_measurements(self) -> None:
        with patch.dict(os.environ, {"KODA_VERBOSE": "0"}), redirect_stdout(self.output):
            with latency_scope(self.clock):
                self.assertTrue(mark_latency("stt_start"))
                mark_latency("stt_done")
                latency_metric("stt_ms", "stt_start", "stt_done")
        self.assertEqual(self.output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
