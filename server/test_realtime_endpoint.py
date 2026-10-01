"""Testes HTTP determinísticos da seleção do fluxo realtime."""

import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import app
from services.conversation import InteractionDecision, InteractionKind
from services.audio_queue import RealtimeAudioQueue
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

    def has_active_command_window(self, _device_id: str) -> bool:
        return False

    def classify(self, _text: str, _device_id: str) -> InteractionDecision:
        self.classify_calls += 1
        return self.decision

    def build_response_from_decision(self, decision: InteractionDecision) -> dict[str, object]:
        self.execute_calls += 1
        assert decision is self.decision
        return dict(self.response)


class _RealtimeAudioDouble:
    def __init__(self) -> None:
        self.create_calls = 0
        self.decisions: list[InteractionDecision] = []

    def create_interaction(self) -> int:
        self.create_calls += 1
        return 42

    def build_stream_audio_response_from_decision(
        self, decision: InteractionDecision, interaction_id: int
    ) -> SimpleNamespace:
        self.decisions.append(decision)
        assert interaction_id == 42
        return SimpleNamespace(audio_segments=())


class RealtimeEndpointTest(TestCase):
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
        self, decision: InteractionDecision, response: dict[str, object] | None = None
    ) -> tuple[object, _ConversationDouble, _RealtimeAudioDouble]:
        conversation = _ConversationDouble(decision, response or {"text": "", "interaction_state": "idle", "actions": []})
        realtime_audio = _RealtimeAudioDouble()
        with (
            patch.object(app, "conversation_manager", conversation),
            patch.object(app, "realtime_audio_service", realtime_audio),
            patch.object(app, "transcribe_audio", return_value=decision.text),
            patch.object(app, "read_wav_upload", AsyncMock(return_value=b"wav")),
            patch.object(app, "validate_wav", return_value=1.0),
            patch.object(app.response_speaker, "speak_response", return_value=False),
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
        self.assertEqual(realtime_audio.create_calls, 0)
        self.assertEqual(realtime_audio.decisions, [])

    def test_wake_returns_sync_attention_without_interaction(self) -> None:
        direct = {"text": "Ok, Rocky", "interaction_state": "attention", "actions": [{"type": "expression", "value": "attention"}]}
        decision = InteractionDecision(InteractionKind.WAKE, "Ok, Rocky", "rocky-01", direct)

        response, conversation, realtime_audio = self.post_with_decision(decision, direct)

        self.assertFalse(response.json()["realtime"])
        self.assertEqual(response.json()["interaction_state"], "attention")
        self.assertEqual(conversation.classify_calls, 1)
        self.assertEqual(realtime_audio.create_calls, 0)

    def test_deterministic_returns_sync_without_interaction(self) -> None:
        direct = {"text": "horas", "interaction_state": "idle", "actions": [{"type": "show_text", "line1": "Agora sao", "line2": "12:00", "duration_ms": 3000}]}
        decision = InteractionDecision(InteractionKind.DETERMINISTIC, "horas", "rocky-01", direct)

        response, conversation, realtime_audio = self.post_with_decision(decision, direct)

        self.assertFalse(response.json()["realtime"])
        self.assertEqual(response.json()["actions"][0]["type"], "show_text")
        self.assertEqual(conversation.execute_calls, 1)
        self.assertEqual(realtime_audio.create_calls, 0)

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
