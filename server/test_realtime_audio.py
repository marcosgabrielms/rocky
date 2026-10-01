"""Testes determinísticos da síntese incremental local por sentença."""

import sys
import types
import unittest
import wave
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

if "piper" not in sys.modules:
    piper_module = types.ModuleType("piper")
    piper_module.PiperVoice = object
    sys.modules["piper"] = piper_module

from services.audio_queue import AudioProductionStatus
from services.conversation import ConversationManager, InteractionKind, LLM_UNAVAILABLE_MESSAGE
from services.llm import ConversationMessage, LLMError, LLMResult
from services.realtime_audio import RealtimeAudioService
from services.tts.realtime import RealtimeTTS


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
    def setUp(self) -> None:
        self._temporary_directory = TemporaryDirectory()
        self.output_directory = Path(self._temporary_directory.name) / "realtime"
        self.clock = FakeClock()

    def tearDown(self) -> None:
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
        service.build_stream_audio_response("Ok, Rocky", device_id)

    def test_one_sentence_generates_one_ordered_segment(self) -> None:
        service, llm, tts = self.build_service(("Olá.",))
        self.wake(service, "one")

        result = service.build_stream_audio_response("Pergunta", "one")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, ["Olá."])
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0])
        self.assertEqual(result.audio_segments[0].sample_rate, 32000)
        self.assertEqual(result.audio_segments[0].channels, 1)
        self.assertEqual(result.audio_segments[0].bits_per_sample, 16)
        self.assertGreater(result.audio_segments[0].duration_ms, 0)
        self.assertGreater(result.audio_segments[0].audio_bytes, 0)

    def test_preclassified_llm_decision_is_not_classified_again(self) -> None:
        service, llm, _ = self.build_service(("Resposta.",))
        service.build_stream_audio_response("Ok, Rocky", "preclassified")
        decision = service._conversation_manager.classify("Pergunta", "preclassified")

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
        wake = service._conversation_manager.classify("Ok, Rocky", "safe-decision")
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

        result = service.build_stream_audio_response("Pergunta", "multiple")

        self.assertEqual(tts.calls, ["Primeira.", "Segunda?", "Terceira!"])
        self.assertEqual([segment.sequence for segment in result.audio_segments], [0, 1, 2])

    def test_broken_chunks_only_synthesize_after_sentence_boundary(self) -> None:
        service, _, tts = self.build_service(("Olá", ", tudo", " bem?"))
        self.wake(service, "chunks")

        service.build_stream_audio_response("Pergunta", "chunks")

        self.assertEqual(tts.calls, ["Olá, tudo bem?"])

    def test_final_residual_is_synthesized(self) -> None:
        service, _, tts = self.build_service(("Resposta sem ponto",))
        self.wake(service, "residual")

        result = service.build_stream_audio_response("Pergunta", "residual")

        self.assertEqual(tts.calls, ["Resposta sem ponto"])
        self.assertEqual(result.conversation.sentences, ("Resposta sem ponto",))

    def test_empty_chunks_do_not_synthesize_audio(self) -> None:
        service, _, tts = self.build_service(("", "  ", "\n"))
        self.wake(service, "empty")

        result = service.build_stream_audio_response("Pergunta", "empty")

        self.assertEqual(tts.calls, [])
        self.assertEqual(result.audio_segments, ())

    def test_tts_failure_on_first_sentence_is_controlled(self) -> None:
        service, llm, tts = self.build_service(("Primeira. Segunda.",), fail_on_call=1)
        self.wake(service, "tts-first-error")

        result = service.build_stream_audio_response("Pergunta", "tts-first-error")

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

        result = service.build_stream_audio_response("Pergunta", "tts-second-error")

        self.assertEqual(tts.calls, ["Primeira.", "Segunda."])
        self.assertTrue(result.tts_failed)
        self.assertEqual([segment.text for segment in result.audio_segments], ["Primeira."])

    def test_llm_error_after_sentence_preserves_existing_audio_without_retry(self) -> None:
        service, llm, tts = self.build_service(("Resposta válida.",), LLMError("stream interrompido"))
        self.wake(service, "partial-llm")

        result = service.build_stream_audio_response("Pergunta", "partial-llm")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, ["Resposta válida."])
        self.assertEqual([segment.text for segment in result.audio_segments], ["Resposta válida."])
        self.assertIsNotNone(result.interaction_id)
        self.assertEqual(
            service.audio_queue.state(result.interaction_id).production_status,
            AudioProductionStatus.FAILED,
        )

    def test_unavailable_free_model_produces_no_audio(self) -> None:
        service, llm, tts = self.build_service((), LLMError("OpenRouter HTTP 429"))
        self.wake(service, "unavailable")

        result = service.build_stream_audio_response("Pergunta", "unavailable")

        self.assertEqual(llm.stream_calls, 1)
        self.assertEqual(tts.calls, [])
        self.assertEqual(result.conversation.response["text"], LLM_UNAVAILABLE_MESSAGE)

    def test_hours_remains_deterministic_without_llm_or_incremental_tts(self) -> None:
        service, llm, tts = self.build_service(("Não deve ser usado.",))
        self.wake(service, "hours")

        with patch("services.conversation.get_current_time", return_value="12:00"):
            result = service.build_stream_audio_response("horas", "hours")

        self.assertEqual(llm.stream_calls, 0)
        self.assertEqual(tts.calls, [])
        self.assertEqual(result.conversation.response["actions"][0]["type"], "show_text")

    def test_new_interaction_preserves_wav_that_is_still_queued(self) -> None:
        service, _, _ = self.build_service(("Primeira.",))
        self.wake(service, "cleanup")
        first = service.build_stream_audio_response("Pergunta", "cleanup")
        preserved_voice = self.output_directory / "pt_BR-jeff-medium.onnx"
        preserved_voice.write_bytes(b"voice")

        self.wake(service, "cleanup")
        second = service.build_stream_audio_response("Outra pergunta", "cleanup")

        self.assertTrue(preserved_voice.exists())
        self.assertTrue(first.audio_segments[0].audio_path.exists())
        self.assertTrue(second.audio_segments[0].audio_path.exists())


if __name__ == "__main__":
    unittest.main()
