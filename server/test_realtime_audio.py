"""Testes determinísticos da síntese incremental local por sentença."""

import sys
import io
import json
import os
import types
import unittest
import wave
from contextlib import redirect_stdout
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

if "piper" not in sys.modules:
    piper_module = types.ModuleType("piper")
    piper_module.PiperVoice = object
    sys.modules["piper"] = piper_module

from services.audio_queue import AudioProductionStatus
from services.conversation import (
    ConversationManager, InteractionKind, InteractionMode, LLM_UNAVAILABLE_MESSAGE, LLM_INTERRUPTED_MESSAGE,
)
from services.llm import ConversationMessage, LLMError, LLMResult, OpenRouterLLMClient
from services.realtime_audio import RealtimeAudioService
from services.tts.realtime import RealtimeTTS
from services.latency import latency_scope


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class StreamingLLM:
    def __init__(self, chunks: tuple[str, ...], error: LLMError | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.stream_calls = 0

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        del message, context
        raise AssertionError("O caminho síncrono não deve ser usado no teste realtime.")

    def stream_response(self, message: str, context: Sequence[ConversationMessage] | None = None):
        del message, context
        self.stream_calls += 1
        yield from self._chunks
        if self._error is not None:
            raise self._error


class FakeTTS:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.calls: list[str] = []
        self._fail_on_call = fail_on_call

    def synthesize_to_wav(self, text: str, output_path: Path) -> None:
        self.calls.append(text)
        if self._fail_on_call == len(self.calls):
            raise RuntimeError("falha Piper simulada")
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 22050)


class RealtimeAudioTest(unittest.TestCase):
    def test_default_ling_reaches_piper_with_contiguous_sequences_without_extra_request(self) -> None:
        from test_realtime_streaming import FakeResponse
        from services.llm.openrouter import DEFAULT_MODEL

        self.assertEqual(DEFAULT_MODEL, "inclusionai/ling-3.1-flash")
        events = (
            {"reasoning": "NEVER SPEAK.", "reasoning_details": [{"text": "INTERNAL."}]},
            {"content": "**Linux** foi criado por Linus Torvalds."},
            {"content": " Ele nasceu na Finlândia."},
        )
        lines = tuple(("data: " + json.dumps({"model": DEFAULT_MODEL, "choices": [{"delta": delta}]}) + "\n").encode()
                      for delta in events) + (b"data: [DONE]\n",)
        with patch.dict(os.environ, {}, clear=True):
            client = OpenRouterLLMClient(api_key="test-key")
        tts = FakeTTS()
        service = RealtimeAudioService(ConversationManager(0, client, clock=self.clock),
                                       RealtimeTTS(tts, self.output_directory))
        logs = io.StringIO()
        with (patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=lines)) as request,
              patch.dict(os.environ, {"KODA_VERBOSE": "1"}),
              redirect_stdout(logs)):
            result = service.build_stream_audio_response("Koda, quem criou o Linux?", "default-ling")
        request.assert_called_once()
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], DEFAULT_MODEL)
        self.assertNotIn("models", payload)
        self.assertEqual(tts.calls, ["Linux foi criado por Linus Torvalds.", "Ele nasceu na Finlândia."])
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0, 1])
        self.assertEqual(service.audio_queue.state(result.interaction_id).production_status, AudioProductionStatus.COMPLETED)
        self.assertEqual(logs.getvalue().count(f"[LLM] routed_model={DEFAULT_MODEL}"), 1)

    def test_default_ling_failure_produces_one_local_audio_and_failed_queue_without_gemma(self) -> None:
        from urllib.error import HTTPError
        from services.llm.openrouter import DEFAULT_MODEL

        with patch.dict(os.environ, {}, clear=True):
            client = OpenRouterLLMClient(api_key="test-key")
        tts = FakeTTS()
        service = RealtimeAudioService(ConversationManager(0, client, clock=self.clock),
                                       RealtimeTTS(tts, self.output_directory))
        error = HTTPError("https://example.test", 429, "failure", None,
                          io.BytesIO(b'{"error":{"code":429,"message":"Rate limit exceeded"}}'))
        with patch("services.llm.openrouter.urlopen", side_effect=error) as request:
            result = service.build_stream_audio_response("Koda, quem criou o Linux?", "ling-failed")
        request.assert_called_once()
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], DEFAULT_MODEL)
        self.assertNotIn("models", payload)
        self.assertEqual(tts.calls, [LLM_UNAVAILABLE_MESSAGE])
        self.assertEqual(result.conversation.response["text"], LLM_UNAVAILABLE_MESSAGE)
        self.assertEqual(result.conversation.response["interaction_state"], "idle")
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0])
        self.assertTrue(result.audio_segments[0].audio_path.exists())
        self.assertEqual(service.audio_queue.state(result.interaction_id).production_status, AudioProductionStatus.FAILED)
        item = service.audio_queue.next_item(result.interaction_id)
        self.assertEqual(item.text, LLM_UNAVAILABLE_MESSAGE)
        service.audio_queue.mark_consumed(result.interaction_id, item.sequence)
        self.assertIsNone(service.audio_queue.next_item(result.interaction_id))

    def test_openrouter_reasoning_never_reaches_sentence_buffer_or_piper(self) -> None:
        from test_realtime_streaming import FakeResponse

        lines = (
            b'data: {"choices":[{"delta":{"reasoning":"INTERNAL REASONING.","reasoning_details":[{"text":"NEVER SPEAK."}]}}]}\n',
            b'data: {"choices":[{"delta":{"content":"**Linux** foi criado por Linus Torvalds."}}]}\n',
            b'data: [DONE]\n',
        )
        llm = OpenRouterLLMClient(api_key="test-key", model="google/gemma-4-31b-it:free")
        manager = ConversationManager(0, llm, clock=self.clock)
        tts = FakeTTS()
        service = RealtimeAudioService(manager, RealtimeTTS(tts, self.output_directory))
        with patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=lines)) as request:
            result = service.build_stream_audio_response("Koda, quem criou o Linux?", "final-only")
        request.assert_called_once()
        self.assertEqual(tts.calls, ["Linux foi criado por Linus Torvalds."])
        self.assertEqual(result.conversation.sentences, ("**Linux** foi criado por Linus Torvalds.",))

    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.output_directory = Path(self._temporary_directory.name) / "realtime"
        self.clock = FakeClock()
        self._verbose = patch.dict(os.environ, {"KODA_VERBOSE": "1"})
        self._verbose.start()

    def tearDown(self) -> None:
        self._verbose.stop()
        self._temporary_directory.cleanup()

    def build_service(
        self, chunks: tuple[str, ...], error: LLMError | None = None, fail_on_call: int | None = None
    ) -> tuple[RealtimeAudioService, StreamingLLM, FakeTTS]:
        llm = StreamingLLM(chunks, error)
        tts = FakeTTS(fail_on_call)
        conversation = ConversationManager(10000, llm, clock=self.clock)
        return RealtimeAudioService(conversation, RealtimeTTS(tts, self.output_directory)), llm, tts

    @staticmethod
    def wake(service: RealtimeAudioService, device_id: str) -> None:
        decision = service._conversation_manager.classify("Koda", device_id, interaction_mode=InteractionMode.WAKE)
        service.build_stream_audio_response_from_decision(decision)

    @staticmethod
    def command(
        service: RealtimeAudioService, text: str, device_id: str, interaction_id: int | None = None,
    ):
        decision = service._conversation_manager.classify(
            text, device_id, interaction_mode=InteractionMode.COMMAND,
        )
        return service.build_stream_audio_response_from_decision(decision, interaction_id)

    def test_one_sentence_generates_one_ordered_segment(self) -> None:
        service, llm, tts = self.build_service(("Olá.",))
        self.wake(service, "one")

        result = self.command(service, "Pergunta", "one")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, ["Olá."])
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0])
        self.assertEqual(result.audio_segments[0].sample_rate, 32000)
        self.assertEqual(result.audio_segments[0].channels, 1)
        self.assertEqual(result.audio_segments[0].bits_per_sample, 16)
        self.assertGreater(result.audio_segments[0].duration_ms, 0)
        self.assertGreater(result.audio_segments[0].audio_bytes, 0)

    def test_markdown_is_cleaned_before_fake_piper_without_changing_backend_text(self) -> None:
        service, llm, tts = self.build_service(("**Linus Torvalds** criou o Linux.",))
        self.wake(service, "markdown")
        result = self.command(service, "Quem criou o Linux?", "markdown")
        self.assertEqual(tts.calls, ["Linus Torvalds criou o Linux."])
        self.assertEqual(result.conversation.response["text"], "**Linus Torvalds** criou o Linux.")
        self.assertEqual(llm.stream_calls, 1)

    def test_skipped_code_sentence_keeps_audio_sequences_contiguous(self) -> None:
        service, llm, tts = self.build_service(("```python\nprint('Codigo.')\n```.", " Resposta: Linux é livre."))
        self.wake(service, "skip-code")
        result = self.command(service, "Pergunta", "skip-code")
        self.assertEqual(tts.calls, ["Linux é livre."])
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0])
        self.assertEqual(llm.stream_calls, 1)

    def test_first_clean_sentence_is_synthesized_before_remaining_stream(self) -> None:
        service, llm, tts = self.build_service(())
        self.wake(service, "incremental-guard")

        def chunks(*_args, **_kwargs):
            llm.stream_calls += 1
            yield "**Primeira**."
            self.assertEqual(tts.calls, ["Primeira."])
            yield " Segunda."

        with patch.object(llm, "stream_response", side_effect=chunks):
            self.command(service, "Pergunta", "incremental-guard")
        self.assertEqual(tts.calls, ["Primeira.", "Segunda."])

    def test_only_code_or_url_produces_no_audio_without_retry(self) -> None:
        for text in ("```python\nprint('Codigo.')\n```", "https://kernel.org"):
            with self.subTest(text=text):
                service, llm, tts = self.build_service((text,))
                self.wake(service, "no-speech")
                result = self.command(service, "Pergunta", "no-speech")
                self.assertEqual(tts.calls, [])
                self.assertEqual(result.audio_segments, ())
                self.assertFalse(result.tts_failed)
                self.assertEqual(llm.stream_calls, 1)

    def test_first_audio_timestamp_precedes_second_sentence_and_generation_end(self) -> None:
        service, llm, _ = self.build_service(())
        self.wake(service, "latency")
        output = io.StringIO()

        def chunks(*_args, **_kwargs):
            llm.stream_calls += 1
            yield "Primeira."
            self.assertIn("kind=event audio_available_ms=", output.getvalue())
            self.assertNotIn("llm_complete_ms=", output.getvalue())
            yield "Segunda."

        with patch.object(llm, "stream_response", side_effect=chunks), redirect_stdout(output), latency_scope():
            result = self.command(service, "Pergunta", "latency")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(len(result.audio_segments), 2)
        self.assertEqual(output.getvalue().count("kind=event audio_available_ms="), 1)
        self.assertIn("kind=duration tts_first_sentence_ms=", output.getvalue())

    def test_preclassified_llm_decision_is_not_classified_again(self) -> None:
        service, llm, _ = self.build_service(("Resposta.",))
        self.wake(service, "preclassified")
        decision = service._conversation_manager.classify(
            "Pergunta", "preclassified", interaction_mode=InteractionMode.COMMAND,
        )

        self.assertEqual(decision.kind, InteractionKind.LLM)
        with patch.object(
            service._conversation_manager,
            "classify",
            side_effect=AssertionError("reclassificação não permitida"),
        ):
            result = service.build_stream_audio_response_from_decision(decision)

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(result.conversation.response["text"], "Resposta.")

    def test_preclassified_wake_or_empty_never_starts_realtime_generation(self) -> None:
        service, llm, tts = self.build_service(("Não deve ser usado.",))
        wake = service._conversation_manager.classify("Koda", "safe-decision", interaction_mode=InteractionMode.WAKE)
        empty = service._conversation_manager.classify("", "empty-decision")

        wake_result = service.build_stream_audio_response_from_decision(wake)
        empty_result = service.build_stream_audio_response_from_decision(empty)

        self.assertEqual(wake.kind, InteractionKind.WAKE)
        self.assertEqual(empty.kind, InteractionKind.EMPTY)
        self.assertEqual(llm.stream_calls, 0)
        self.assertEqual(tts.calls, [])
        self.assertIsNone(wake_result.interaction_id)
        self.assertIsNone(empty_result.interaction_id)

    def test_multiple_sentences_keep_sequence_order(self) -> None:
        service, _, tts = self.build_service(("Primeira. Segunda? Terceira!",))
        self.wake(service, "multiple")

        result = self.command(service, "Pergunta", "multiple")

        self.assertEqual(tts.calls, ["Primeira.", "Segunda?", "Terceira!"])
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0, 1, 2])

    def test_broken_chunks_only_synthesize_after_sentence_boundary(self) -> None:
        service, _, tts = self.build_service(("Olá", ", tudo", " bem?"))
        self.wake(service, "chunks")

        self.command(service, "Pergunta", "chunks")

        self.assertEqual(tts.calls, ["Olá, tudo bem?"])

    def test_final_residual_is_synthesized(self) -> None:
        service, _, tts = self.build_service(("Resposta sem ponto",))
        self.wake(service, "residual")

        result = self.command(service, "Pergunta", "residual")

        self.assertEqual(tts.calls, ["Resposta sem ponto"])
        self.assertEqual(result.conversation.sentences, ("Resposta sem ponto",))

    def test_empty_generation_synthesizes_only_local_failure_audio(self) -> None:
        service, _, tts = self.build_service(("", "  ", "\n"))
        self.wake(service, "empty")

        result = self.command(service, "Pergunta", "empty")

        self.assertEqual(tts.calls, [LLM_UNAVAILABLE_MESSAGE])
        self.assertEqual(len(result.audio_segments), 1)

    def test_tts_failure_on_first_sentence_is_controlled(self) -> None:
        service, llm, tts = self.build_service(("Primeira. Segunda.",), fail_on_call=1)
        self.wake(service, "tts-first-error")

        result = self.command(service, "Pergunta", "tts-first-error")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, ["Primeira."])
        self.assertTrue(result.tts_failed)
        self.assertEqual(result.audio_segments, ())
        self.assertIsNotNone(result.interaction_id)
        self.assertEqual(
            service.audio_queue.state(result.interaction_id).production_status,
            AudioProductionStatus.FAILED,
        )

    def test_tts_failure_preserves_previous_audio_and_stops_audio_pipeline(self) -> None:
        service, _, tts = self.build_service(("Primeira. Segunda. Terceira.",), fail_on_call=2)
        self.wake(service, "tts-second-error")

        result = self.command(service, "Pergunta", "tts-second-error")

        self.assertEqual(tts.calls, ["Primeira.", "Segunda."])
        self.assertTrue(result.tts_failed)
        self.assertEqual([segment.text for segment in result.audio_segments], ["Primeira."])

    def test_llm_error_after_sentence_preserves_existing_audio_without_retry(self) -> None:
        service, llm, tts = self.build_service(("Resposta válida.",), LLMError("stream interrompido"))
        self.wake(service, "partial-llm")

        result = self.command(service, "Pergunta", "partial-llm")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, ["Resposta válida.", LLM_INTERRUPTED_MESSAGE])
        self.assertEqual([segment.text for segment in result.audio_segments], ["Resposta válida.", LLM_INTERRUPTED_MESSAGE])
        self.assertIsNotNone(result.interaction_id)
        self.assertEqual(
            service.audio_queue.state(result.interaction_id).production_status,
            AudioProductionStatus.FAILED,
        )

    def test_unavailable_free_model_produces_one_local_error_audio(self) -> None:
        service, llm, tts = self.build_service((), LLMError("OpenRouter HTTP 429"))
        self.wake(service, "unavailable")

        result = self.command(service, "Pergunta", "unavailable")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, [LLM_UNAVAILABLE_MESSAGE])
        self.assertEqual(result.conversation.response["text"], LLM_UNAVAILABLE_MESSAGE)
        self.assertEqual(service.audio_queue.state(result.interaction_id).production_status, AudioProductionStatus.FAILED)

    def test_unexpected_network_errors_finish_production_without_retry_or_secret_logs(self) -> None:
        from http.client import IncompleteRead
        from ssl import SSLError
        for error in (ConnectionResetError("SECRET"), IncompleteRead(b"SECRET"),
                      SSLError("SECRET"), RuntimeError("SECRET")):
            with self.subTest(error=type(error).__name__):
                service, llm, tts = self.build_service((), error)
                self.wake(service, "network-error")
                interaction_id = service.create_interaction()
                logs = io.StringIO()
                with redirect_stdout(logs):
                    result = self.command(service, "Pergunta", "network-error", interaction_id)
                self.assertEqual(llm.stream_calls, 1)
                self.assertEqual(tts.calls, [LLM_UNAVAILABLE_MESSAGE])
                self.assertTrue(result.conversation.stream_failed)
                self.assertEqual(service.audio_queue.state(interaction_id).production_status, AudioProductionStatus.FAILED)
                self.assertNotIn("SECRET", logs.getvalue())

    def test_session_setup_error_is_terminal(self) -> None:
        service, _, _ = self.build_service(("Resposta.",))
        self.wake(service, "setup-error")
        interaction_id = service.create_interaction()
        with patch.object(service._realtime_tts, "begin_interaction", side_effect=OSError("SECRET")):
            result = self.command(service, "Pergunta", "setup-error", interaction_id)
        self.assertTrue(result.tts_failed)
        self.assertEqual(service.audio_queue.state(interaction_id).production_status, AudioProductionStatus.FAILED)

    def test_stream_deadline_produces_error_audio_and_failed_queue_without_retry(self) -> None:
        from types import SimpleNamespace
        from test_realtime_streaming import FakeResponse
        clock = SimpleNamespace(now=0.0)

        class KeepaliveResponse(FakeResponse):
            def __iter__(self):
                while True:
                    clock.now += 61
                    yield b": keepalive\n"

        client = OpenRouterLLMClient(api_key="test-key", model="google/gemma-4-31b-it:free")
        manager = ConversationManager(0, client, clock=self.clock)
        tts = FakeTTS()
        service = RealtimeAudioService(manager, RealtimeTTS(tts, self.output_directory))
        with (patch("services.llm.openrouter.time.monotonic", side_effect=lambda: clock.now),
              patch("services.llm.openrouter.Timer"),
              patch("services.llm.openrouter.urlopen", return_value=KeepaliveResponse()) as request):
            result = service.build_stream_audio_response("Koda, quem criou o Linux?", "deadline")
        request.assert_called_once()
        self.assertEqual(tts.calls, [LLM_UNAVAILABLE_MESSAGE])
        self.assertEqual(service.audio_queue.state(result.interaction_id).production_status, AudioProductionStatus.FAILED)

    def test_base_exception_propagates_but_queue_is_not_left_active(self) -> None:
        for error_type in (KeyboardInterrupt, SystemExit):
            with self.subTest(error=error_type.__name__, source="legacy"):
                service, _, _ = self.build_service((), error_type())
                self.wake(service, "interrupt")
                interaction_id = service.create_interaction()
                with self.assertRaises(error_type):
                    self.command(service, "Pergunta", "interrupt", interaction_id)
                self.assertEqual(service.audio_queue.state(interaction_id).production_status, AudioProductionStatus.FAILED)

            class InterruptingCoreAdapter:
                def build_stream_response(self, *_args, **_kwargs) -> object:
                    raise error_type()

            with self.subTest(error=error_type.__name__, source="core"):
                service, _, _ = self.build_service(())
                interaction_id = service.create_interaction()
                with self.assertRaises(error_type):
                    service.build_stream_audio_response_from_core_decision(
                        InterruptingCoreAdapter(), object(), interaction_id,
                    )
                self.assertEqual(service.audio_queue.state(interaction_id).production_status, AudioProductionStatus.FAILED)

    def test_partial_tts_artifact_is_removed_on_failure(self) -> None:
        service, _, tts = self.build_service(("Resposta.",))
        self.wake(service, "partial-file")

        def fail(text, path):
            path.write_bytes(b"partial wav")
            raise OSError("falha simulada")

        with patch.object(tts, "synthesize_to_wav", side_effect=fail):
            result = self.command(service, "Pergunta", "partial-file")
        self.assertTrue(result.tts_failed)
        self.assertEqual(list(self.output_directory.glob("rocky_rt_*.wav")), [])

    def test_new_interaction_removes_old_orphan_but_preserves_recent_audio_and_models(self) -> None:
        service, _, _ = self.build_service(())
        self.output_directory.mkdir()
        orphan = self.output_directory / "rocky_rt_9999_000.wav"
        recent = self.output_directory / "rocky_rt_9999_001.wav"
        model = self.output_directory / "pt_BR-jeff-medium.onnx"
        orphan.write_bytes(b"old wav")
        recent.write_bytes(b"new wav")
        model.write_bytes(b"voice")
        current = orphan.stat().st_mtime + 301
        from types import SimpleNamespace
        real_stat = Path.stat

        def stat(path, *args, **kwargs):
            if path == recent:
                return SimpleNamespace(st_mtime=current)
            return real_stat(path, *args, **kwargs)

        with patch("services.tts.realtime.time.time", return_value=current), patch.object(Path, "stat", stat):
            service.create_interaction()
        self.assertFalse(orphan.exists())
        self.assertTrue(recent.exists())
        self.assertEqual(model.read_bytes(), b"voice")

    def test_hours_remains_deterministic_without_llm_or_incremental_tts(self) -> None:
        service, llm, tts = self.build_service(("Não deve ser usado.",))
        self.wake(service, "hours")

        with patch("services.conversation.get_current_time", return_value="12:00"):
            result = self.command(service, "horas", "hours")

        self.assertEqual(llm.stream_calls, 0)
        self.assertEqual(tts.calls, [])
        self.assertEqual(result.conversation.response["actions"][0]["type"], "show_text")

    def test_deterministic_response_uses_the_realtime_audio_queue_without_llm(self) -> None:
        service, llm, tts = self.build_service(("Não deve ser usado.",))
        response = {
            "text": "hora",
            "interaction_state": "idle",
            "actions": [{"type": "show_text", "line1": "Agora sao", "line2": "12:00", "duration_ms": 3000}],
        }

        interaction_id = service.queue_direct_response(response, "hora")

        self.assertIsNotNone(interaction_id)
        self.assertEqual(llm.stream_calls, 0)
        self.assertEqual(tts.calls, ["Agora sao 12:00"])
        item = service.audio_queue.next_item(interaction_id)
        self.assertIsNotNone(item)
        self.assertEqual(item.text, "Agora sao 12:00")
        self.assertEqual(service.audio_queue.state(interaction_id).production_status, AudioProductionStatus.COMPLETED)

    def test_new_interaction_preserves_wav_that_is_still_queued(self) -> None:
        service, _, _ = self.build_service(("Primeira.",))
        self.wake(service, "cleanup")
        first = self.command(service, "Pergunta", "cleanup")
        preserved_voice = self.output_directory / "pt_BR-jeff-medium.onnx"
        preserved_voice.write_bytes(b"voice")

        self.wake(service, "cleanup")
        second = self.command(service, "Outra pergunta", "cleanup")

        self.assertTrue(preserved_voice.exists())
        self.assertTrue(first.audio_segments[0].audio_path.exists())
        self.assertTrue(second.audio_segments[0].audio_path.exists())


if __name__ == "__main__":
    unittest.main()
