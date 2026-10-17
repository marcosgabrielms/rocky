"""Testes da camada textual reutilizável de interação do Koda."""

import unittest

from koda_core import CoreContextMessage, CoreDecision, InteractionResult
from services.interaction import KodaInteractionService


class _AdapterDouble:
    def __init__(self, decision: CoreDecision) -> None:
        self.decision = decision
        self.classify_calls: list[tuple[object, dict[str, object]]] = []
        self.response_calls: list[CoreDecision] = []
        self.direct_calls: list[CoreDecision] = []
        self.stream_calls: list[tuple[CoreDecision, object]] = []

    def classify(self, request, **kwargs):
        self.classify_calls.append((request, kwargs))
        return self.decision

    def build_response(self, decision):
        self.response_calls.append(decision)
        return decision.direct_result

    def build_direct_result(self, decision):
        self.direct_calls.append(decision)
        return decision.direct_result

    def build_stream_response(self, decision, *, on_sentence=None):
        self.stream_calls.append((decision, on_sentence))
        return "streamed"


class KodaInteractionServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.direct_result = InteractionResult(
            text="Volume ajustado para 50%.",
            interaction_state="idle",
            actions=[{"type": "set_volume", "percent": 50}],
        )
        self.decision = CoreDecision(
            kind="deterministic",
            text="volume em 50%",
            device_id="device-01",
            direct_result=self.direct_result,
            context=(CoreContextMessage(role="user", content="pergunta anterior"),),
        )
        self.adapter = _AdapterDouble(self.decision)
        self.service = KodaInteractionService(self.adapter)

    def test_classify_builds_and_preserves_interaction_request(self) -> None:
        result = self.service.classify(
            user_id="user-01",
            device_id="device-01",
            input_type="voice_transcript",
            text="volume em 50%",
            received_at=12.5,
            interaction_mode="command",
            low_confidence=True,
            metadata={"source": "test"},
        )

        request, kwargs = self.adapter.classify_calls[0]
        self.assertIs(result, self.decision)
        self.assertEqual(request.user_id, "user-01")
        self.assertEqual(request.device_id, "device-01")
        self.assertEqual(request.input_type, "voice_transcript")
        self.assertEqual(request.text, "volume em 50%")
        self.assertEqual(request.metadata, {"source": "test"})
        self.assertEqual(kwargs, {"received_at": 12.5, "interaction_mode": "command", "low_confidence": True})

    def test_direct_response_preserves_actions_without_reclassification(self) -> None:
        classified = self.service.classify(
            user_id="user", device_id="device-01", input_type="text", text="volume em 50%",
        )
        result = self.service.build_direct_response(classified)

        self.assertIs(result, self.direct_result)
        self.assertEqual(result.actions, [{"type": "set_volume", "percent": 50}])
        self.assertEqual(len(self.adapter.classify_calls), 1)
        self.assertEqual(self.adapter.direct_calls, [self.decision])

    def test_normal_llm_response_keeps_decision_context_and_device(self) -> None:
        llm_decision = CoreDecision(
            kind="llm",
            text="E de onde ele é?",
            device_id="device-context",
            context=(CoreContextMessage(role="assistant", content="Linus Torvalds."),),
        )
        adapter = _AdapterDouble(llm_decision)
        service = KodaInteractionService(adapter)

        classified = service.classify(
            user_id="user", device_id="device-context", input_type="text", text=llm_decision.text,
        )
        service.build_response(classified)

        self.assertEqual(classified.device_id, "device-context")
        self.assertEqual(classified.context, llm_decision.context)
        self.assertEqual(adapter.response_calls, [llm_decision])
        self.assertEqual(len(adapter.classify_calls), 1)

    def test_streaming_delegates_without_reclassification(self) -> None:
        callback = object()
        result = self.service.build_stream_response(self.decision, on_sentence=callback)

        self.assertEqual(result, "streamed")
        self.assertEqual(self.adapter.stream_calls, [(self.decision, callback)])
        self.assertEqual(self.adapter.classify_calls, [])

    def test_service_has_no_command_session_or_audio_dependencies(self) -> None:
        self.assertFalse(hasattr(self.service, "command_session"))
        self.assertFalse(hasattr(self.service, "audio_queue"))


if __name__ == "__main__":
    unittest.main()
