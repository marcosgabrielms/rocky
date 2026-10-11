"""Testes HTTP determinísticos da seleção do fluxo realtime."""

import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

import app
from koda_core import ConversationCoreAdapter
from services.conversation import ConversationManager, InteractionDecision, InteractionKind, InteractionMode
from services.audio_queue import RealtimeAudioQueue
from services.llm.base import ConversationMessage
from services.stt import TranscriptionResult
from services.tts.realtime import RealtimeAudioSegment


class _ImmediateThread:
    def __init__(self, target, **_kwargs) -> None:
        self._target = target

    def start(self) -> None:
        self._target()


class _ConversationDouble:
    def __init__(self, decision: InteractionDecision, response: dict[str, object]) -> None:
        self.decision = decision
        self.response = response
        self.classify_calls = 0
        self.execute_calls = 0
        self.received_at: float | None = None
        self.classify_arguments: tuple[str, str, InteractionMode | None, bool] | None = None

    def capture_request_time(self) -> float:
        return 0.0

    def has_active_command_window(self, _device_id: str, *, received_at: float | None = None) -> bool:
        return False

    def classify(self, _text: str, _device_id: str, *, received_at: float | None = None,
                 interaction_mode: InteractionMode | None = None, low_confidence: bool = False) -> InteractionDecision:
        self.classify_calls += 1
        self.received_at = received_at
        self.classify_arguments = (_text, _device_id, interaction_mode, low_confidence)
        return self.decision

    def build_response_from_decision(self, decision: InteractionDecision) -> dict[str, object]:
        self.execute_calls += 1
        assert decision == self.decision
        return dict(self.response)


class _RealtimeAudioDouble:
    def __init__(self) -> None:
        self.create_calls = 0
        self.decisions: list[InteractionDecision] = []
        self.direct_responses: list[tuple[dict[str, object], str]] = []
        self.direct_interaction_id: int | None = None
        self.audio_queue = Mock(spec=RealtimeAudioQueue)

    def create_interaction(self) -> int:
        self.create_calls += 1
        return 42

    def build_stream_audio_response_from_decision(
        self, decision: InteractionDecision, interaction_id: int
    ) -> SimpleNamespace:
        self.decisions.append(decision)
        assert interaction_id == 42
        return SimpleNamespace(audio_segments=())

    def queue_direct_response(self, response: dict[str, object], request_text: str) -> int | None:
        self.direct_responses.append((dict(response), request_text))
        return self.direct_interaction_id


class RealtimeEndpointTest(TestCase):
    @staticmethod
    def core_adapter(conversation: object) -> ConversationCoreAdapter:
        return ConversationCoreAdapter(
            conversation,
            decision_factory=InteractionDecision,
            context_message_factory=ConversationMessage,
            kind_factory=InteractionKind,
        )

    def test_health_works_without_llm_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            response = TestClient(app.app).get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_thread_start_and_escaped_producer_errors_are_terminal(self) -> None:
        from services.audio_queue import AudioProductionStatus
        for thread_start_error in (False, True):
            with self.subTest(thread_start_error=thread_start_error), TemporaryDirectory() as directory:
                queue = RealtimeAudioQueue(Path(directory))
                producer = Mock(side_effect=ConnectionResetError("SECRET"))
                service = SimpleNamespace(audio_queue=queue, create_interaction=queue.create_interaction,
                                          build_stream_audio_response_from_decision=producer)
                decision = InteractionDecision(InteractionKind.LLM, "Pergunta", "test")
                conversation = _ConversationDouble(decision, {})
                thread = Mock()
                thread.return_value.start.side_effect = RuntimeError("SECRET")
                with (patch.object(app, "realtime_audio_service", service),
                      patch.object(app, "conversation_manager", conversation),
                      patch.object(app, "conversation_core_adapter", self.core_adapter(conversation)),
                      patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
                      patch.object(app, "validate_wav", return_value=1.0),
                      patch.object(app, "transcribe_audio", return_value="Pergunta"),
                      patch.object(app, "Thread", thread if thread_start_error else _ImmediateThread)):
                    response = TestClient(app.app).post("/transcribe/realtime",
                        files={"file": ("speech.wav", b"wav", "audio/wav")})
                    interaction_id = response.json()["interaction_id"]
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(queue.state(interaction_id).production_status, AudioProductionStatus.FAILED)
                    self.assertEqual(TestClient(app.app).get(f"/realtime/audio/{interaction_id}/next").json(), {"status": "failed"})

    def test_llm_failure_serves_local_error_wav_then_failed_after_ack(self) -> None:
        from services.llm import LLMError
        from services.realtime_audio import RealtimeAudioService
        from services.tts.realtime import RealtimeTTS
        from test_realtime_audio import FakeTTS, StreamingLLM
        with TemporaryDirectory() as directory:
            llm = StreamingLLM((), LLMError("OpenRouter HTTP 429"))
            conversation = ConversationManager(0, llm)
            tts = FakeTTS()
            service = RealtimeAudioService(conversation, RealtimeTTS(tts, Path(directory)))
            with (patch.object(app, "conversation_manager", conversation),
                  patch.object(app, "conversation_core_adapter", self.core_adapter(conversation)),
                  patch.object(app, "realtime_audio_service", service),
                  patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
                  patch.object(app, "validate_wav", return_value=1.0),
                  patch.object(app, "transcribe_audio", return_value="Koda, uma pergunta?"),
                  patch.object(app, "Thread", _ImmediateThread)):
                client = TestClient(app.app)
                response = client.post("/transcribe/realtime", files={"file": ("speech.wav", b"wav", "audio/wav")})
                interaction_id = response.json()["interaction_id"]
                audio = client.get(f"/realtime/audio/{interaction_id}/next")
                self.assertEqual(audio.status_code, 200)
                self.assertEqual(audio.content[:4], b"RIFF")
                self.assertEqual(audio.headers["x-rocky-sequence"], "0")
                self.assertEqual(client.post(f"/realtime/audio/{interaction_id}/0/consumed").status_code, 200)
                self.assertEqual(client.get(f"/realtime/audio/{interaction_id}/next").json(), {"status": "failed"})
            self.assertEqual(llm.stream_calls, 1)
            self.assertEqual(len(tts.calls), 1)

    def test_explicit_command_mode_survives_slow_stt_in_both_routes(self) -> None:
        for endpoint in ("/transcribe", "/transcribe/realtime"):
            with self.subTest(endpoint=endpoint):
                clock = SimpleNamespace(now=0.0)
                llm = SimpleNamespace(ask=lambda *_args, **_kwargs: self.fail("horas nao chama LLM"))
                conversation = ConversationManager(0, llm, clock=lambda: clock.now)
                conversation.build_response("Koda", "physical", interaction_mode=InteractionMode.WAKE)

                def slow_stt(*_args):
                    clock.now = 60.0
                    return "Que horas são?"

                with (
                    patch.object(app, "conversation_manager", conversation),
                    patch.object(app, "conversation_core_adapter", self.core_adapter(conversation)),
                    patch.object(app, "transcribe_audio", side_effect=slow_stt) as stt,
                    patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
                    patch.object(app, "validate_wav", return_value=1.0),
                    patch.object(app.response_speaker, "speak_response", return_value=False),
                    patch.object(app.realtime_audio_service, "queue_direct_response", return_value=None),
                    patch("services.conversation.get_current_time", return_value="12:00"),
                ):
                    response = TestClient(app.app).post(
                        endpoint, headers={"X-Rocky-Device": "physical", "X-Rocky-Interaction-Mode": "command"},
                        files={"file": ("speech.wav", b"wav", "audio/wav")},
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["actions"][0]["line2"], "12:00")
                self.assertFalse(stt.call_args.args[2])

    def test_invalid_interaction_mode_is_rejected_before_stt(self) -> None:
        with patch.object(app, "transcribe_audio") as stt:
            response = TestClient(app.app).post("/transcribe/realtime",
                                               headers={"X-Rocky-Interaction-Mode": "invalid"})
        self.assertEqual(response.status_code, 422)
        stt.assert_not_called()

    @staticmethod
    def queue_segment(directory: Path, sequence: int, text: str) -> RealtimeAudioSegment:
        path = directory / f"segment_{sequence}_{text}.wav"
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(32000)
            wav_file.writeframes(b"\x00\x00" * 32)
        return RealtimeAudioSegment(sequence, text, path, 32000, 1, 16, 1, path.stat().st_size)

    def post_with_decision(
        self, decision: InteractionDecision, response: dict[str, object] | None = None,
        direct_interaction_id: int | None = None,
    ) -> tuple[object, _ConversationDouble, _RealtimeAudioDouble]:
        conversation = _ConversationDouble(decision, response or {"text": "", "interaction_state": "idle", "actions": []})
        realtime_audio = _RealtimeAudioDouble()
        realtime_audio.direct_interaction_id = direct_interaction_id
        with (
            patch.object(app, "conversation_manager", conversation),
            patch.object(app, "conversation_core_adapter", self.core_adapter(conversation)),
            patch.object(app, "realtime_audio_service", realtime_audio),
            patch.object(app, "transcribe_audio", return_value=decision.text),
            patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
            patch.object(app, "validate_wav", return_value=1.0),
            patch.object(app, "Thread", _ImmediateThread),
        ):
            response_http = TestClient(app.app).post(
                "/transcribe/realtime",
                files={"file": ("speech.wav", b"wav", "audio/wav")},
            )
        return response_http, conversation, realtime_audio

    def test_empty_returns_sync_without_interaction_or_thread(self) -> None:
        decision = InteractionDecision(InteractionKind.EMPTY, "", "rocky-01", {"text": "", "interaction_state": "idle", "actions": []})

        response, conversation, realtime_audio = self.post_with_decision(decision)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["realtime"])
        self.assertIsNone(response.json()["interaction_id"])
        self.assertEqual(conversation.classify_calls, 1)
        self.assertEqual(conversation.execute_calls, 1)
        self.assertEqual(conversation.received_at, 0.0)
        self.assertEqual(realtime_audio.create_calls, 0)
        self.assertEqual(realtime_audio.decisions, [])

    def test_wake_returns_sync_attention_without_interaction(self) -> None:
        direct = {"text": "Koda", "interaction_state": "attention", "actions": [{"type": "expression", "value": "attention"}]}
        decision = InteractionDecision(InteractionKind.WAKE, "Koda", "rocky-01", direct)

        response, conversation, realtime_audio = self.post_with_decision(decision, direct)

        self.assertFalse(response.json()["realtime"])
        self.assertEqual(response.json()["interaction_state"], "attention")
        self.assertEqual(conversation.classify_calls, 1)
        self.assertEqual(realtime_audio.create_calls, 0)

    def test_realtime_creates_core_request_and_uses_direct_result(self) -> None:
        direct = {"text": "Koda", "interaction_state": "attention", "actions": []}
        decision = InteractionDecision(InteractionKind.WAKE, "Koda", "koda-core", direct)
        conversation = _ConversationDouble(decision, direct)
        realtime_audio = _RealtimeAudioDouble()
        adapter = Mock(wraps=self.core_adapter(conversation))
        transcription = TranscriptionResult("Koda", has_speech=True, low_confidence=True)

        with (
            patch.object(app, "conversation_manager", conversation),
            patch.object(app, "conversation_core_adapter", adapter),
            patch.object(app, "realtime_audio_service", realtime_audio),
            patch.object(app, "transcribe_audio", return_value=transcription),
            patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
            patch.object(app, "validate_wav", return_value=1.0),
        ):
            response = TestClient(app.app).post(
                "/transcribe/realtime",
                headers={
                    "X-Rocky-Device": "koda-core",
                    "X-Rocky-Interaction-Mode": "command",
                },
                files={"file": ("speech.wav", b"wav", "audio/wav")},
            )

        request = adapter.classify.call_args.args[0]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(request.user_id, "local-user")
        self.assertEqual(request.device_id, "koda-core")
        self.assertEqual(request.input_type, "voice_transcript")
        self.assertEqual(request.text, "Koda")
        self.assertEqual(conversation.classify_arguments,
                         ("Koda", "koda-core", InteractionMode.COMMAND, True))
        adapter.build_direct_result.assert_called_once()
        adapter.to_conversation_decision.assert_not_called()
        self.assertEqual(conversation.classify_calls, 1)

    def test_realtime_continues_llm_with_rebuilt_decision_without_reclassification(self) -> None:
        context = (ConversationMessage(role="user", content="Quem criou o Linux?"),)
        decision = InteractionDecision(InteractionKind.LLM, "E de onde ele e?", "koda-core", context=context)
        conversation = _ConversationDouble(decision, {})
        realtime_audio = _RealtimeAudioDouble()
        adapter = Mock(wraps=self.core_adapter(conversation))

        with (
            patch.object(app, "conversation_manager", conversation),
            patch.object(app, "conversation_core_adapter", adapter),
            patch.object(app, "realtime_audio_service", realtime_audio),
            patch.object(app, "transcribe_audio", return_value=decision.text),
            patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
            patch.object(app, "validate_wav", return_value=1.0),
            patch.object(app, "Thread", _ImmediateThread),
        ):
            response = TestClient(app.app).post(
                "/transcribe/realtime",
                headers={"X-Rocky-Device": "koda-core"},
                files={"file": ("speech.wav", b"wav", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        adapter.classify.assert_called_once()
        adapter.build_direct_result.assert_not_called()
        adapter.to_conversation_decision.assert_called_once()
        self.assertEqual(conversation.classify_calls, 1)
        self.assertEqual(realtime_audio.decisions, [decision])
        self.assertEqual(realtime_audio.decisions[0].context, context)

    def test_deterministic_uses_realtime_interaction_without_local_playback(self) -> None:
        direct = {"text": "horas", "interaction_state": "idle", "actions": [{"type": "show_text", "line1": "Agora sao", "line2": "12:00", "duration_ms": 3000}]}
        decision = InteractionDecision(InteractionKind.DETERMINISTIC, "horas", "rocky-01", direct)

        with patch.object(app.response_speaker, "speak_response") as local_playback:
            response, conversation, realtime_audio = self.post_with_decision(decision, direct, 84)

        self.assertTrue(response.json()["realtime"])
        self.assertEqual(response.json()["interaction_id"], 84)
        self.assertEqual(response.json()["actions"][0]["type"], "show_text")
        self.assertEqual(conversation.execute_calls, 1)
        self.assertEqual(realtime_audio.create_calls, 0)
        self.assertEqual(realtime_audio.direct_responses, [(direct, "horas")])
        local_playback.assert_not_called()

    def test_llm_creates_one_interaction_and_passes_the_same_decision_to_thread(self) -> None:
        decision = InteractionDecision(InteractionKind.LLM, "Pergunta aberta", "rocky-01")

        response, conversation, realtime_audio = self.post_with_decision(decision)

        self.assertTrue(response.json()["realtime"])
        self.assertEqual(response.json()["interaction_id"], 42)
        self.assertEqual(response.json()["interaction_state"], "thinking")
        self.assertEqual(conversation.classify_calls, 1)
        self.assertEqual(conversation.execute_calls, 0)
        self.assertEqual(realtime_audio.create_calls, 1)
        self.assertEqual(realtime_audio.decisions, [decision])

    def test_next_reports_pending_done_failed_cancelled_and_unknown(self) -> None:
        with TemporaryDirectory() as directory:
            queue = RealtimeAudioQueue(Path(directory))
            pending = queue.create_interaction()
            done = queue.create_interaction()
            failed = queue.create_interaction()
            cancelled = queue.create_interaction()
            queue.mark_production_done(done)
            queue.mark_production_failed(failed)
            queue.cancel(cancelled)
            service = SimpleNamespace(audio_queue=queue)
            with patch.object(app, "realtime_audio_service", service):
                client = TestClient(app.app)
                self.assertEqual(client.get(f"/realtime/audio/{pending}/next?timeout_ms=0").json()["status"], "pending")
                self.assertEqual(client.get(f"/realtime/audio/{done}/next").json()["status"], "done")
                self.assertEqual(client.get(f"/realtime/audio/{failed}/next").json()["status"], "failed")
                self.assertEqual(client.get(f"/realtime/audio/{cancelled}/next").json()["status"], "cancelled")
                self.assertEqual(client.get("/realtime/audio/999/next").status_code, 404)

    def test_ready_ack_order_and_interaction_isolation(self) -> None:
        with TemporaryDirectory() as directory:
            audio_directory = Path(directory)
            queue = RealtimeAudioQueue(audio_directory)
            first = queue.create_interaction()
            second = queue.create_interaction()
            first_item = queue.enqueue(first, self.queue_segment(audio_directory, 0, "first"))
            second_item = queue.enqueue(second, self.queue_segment(audio_directory, 0, "second"))
            service = SimpleNamespace(audio_queue=queue)
            with patch.object(app, "realtime_audio_service", service):
                client = TestClient(app.app)
                ready = client.get(f"/realtime/audio/{first}/next")
                self.assertEqual(ready.status_code, 200)
                self.assertEqual(ready.headers["content-type"], "audio/wav")
                self.assertEqual(ready.headers["x-rocky-interaction"], str(first))
                self.assertEqual(ready.headers["x-rocky-sequence"], "0")
                self.assertEqual(ready.content[:4], b"RIFF")
                self.assertEqual(queue.next_item(second).audio_path, second_item.audio_path)
                self.assertEqual(client.post(f"/realtime/audio/{first}/1/consumed").status_code, 409)
                self.assertEqual(client.post(f"/realtime/audio/{first}/0/consumed").json()["status"], "consumed")
                self.assertFalse(first_item.audio_path.exists())
                self.assertEqual(client.post(f"/realtime/audio/{first}/0/consumed").status_code, 409)


if __name__ == "__main__":
    import unittest

    unittest.main()
