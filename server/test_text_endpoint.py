"""Testes HTTP da entrada textual independente de dispositivo."""

from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import app
from koda_core import CoreContextMessage, CoreDecision, InteractionResult, UsageRecord
from services.conversation import InteractionMode


class TextEndpointTest(TestCase):
    def post_with_service(
        self, decision: CoreDecision, result: InteractionResult
    ) -> tuple[object, Mock, Mock, Mock, Mock, Mock]:
        interaction_service = Mock()
        interaction_service.classify.return_value = decision
        interaction_service.build_response.return_value = result
        command_session = Mock()

        with (
            patch.object(app, "interaction_service", interaction_service),
            patch.object(app, "command_session", command_session),
            patch.object(app, "transcribe_audio") as transcribe_audio,
            patch.object(app.response_speaker, "speak_response") as speak_response,
            patch.object(app, "realtime_audio_service") as realtime_audio_service,
        ):
            response = TestClient(app.app).post(
                "/interact",
                json={"text": " volume em 50% ", "device_id": " text-client "},
            )

        return (
            response,
            interaction_service,
            command_session,
            transcribe_audio,
            speak_response,
            realtime_audio_service,
        )

    def test_uses_interaction_service_once_for_direct_response_and_keeps_actions(self) -> None:
        result = InteractionResult(
            text="Volume ajustado para 50%.",
            interaction_state="idle",
            actions=[{"type": "set_volume", "percent": 50}],
        )
        decision = CoreDecision(
            kind="deterministic",
            text="volume em 50%",
            device_id="text-client",
            direct_result=result,
        )

        response, service, session, transcribe_audio, speak_response, realtime_audio = self.post_with_service(decision, result)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "text": "Volume ajustado para 50%.",
            "interaction_state": "idle",
            "actions": [{"type": "set_volume", "percent": 50}],
        })
        service.classify.assert_called_once_with(
            user_id="local-user",
            device_id="text-client",
            input_type="text",
            text="volume em 50%",
            interaction_mode=InteractionMode.COMMAND,
            low_confidence=False,
        )
        service.build_response.assert_called_once_with(decision)
        self.assertEqual(session.method_calls, [])
        transcribe_audio.assert_not_called()
        speak_response.assert_not_called()
        realtime_audio.assert_not_called()

    def test_llm_response_preserves_provider_model_usage_and_context_decision(self) -> None:
        result = InteractionResult(
            text="Linus Torvalds Ã© finlandÃªs.",
            interaction_state="idle",
            provider="chatgpt",
            model="gpt-5.6-terra",
            usage=UsageRecord(provider="chatgpt", model="gpt-5.6-terra", input_tokens=7, output_tokens=5, total_tokens=12),
        )
        decision = CoreDecision(
            kind="llm",
            text="E de onde ele Ã©?",
            device_id="text-client",
            context=(CoreContextMessage(role="assistant", content="Linus Torvalds criou o Linux."),),
        )

        response, service, session, transcribe_audio, speak_response, realtime_audio = self.post_with_service(decision, result)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "Linus Torvalds Ã© finlandÃªs.")
        self.assertEqual(response.json()["provider"], "chatgpt")
        self.assertEqual(response.json()["model"], "gpt-5.6-terra")
        self.assertEqual(response.json()["usage"]["total_tokens"], 12)
        self.assertEqual(service.classify.call_count, 1)
        self.assertEqual(service.classify.call_args.kwargs["device_id"], "text-client")
        service.build_response.assert_called_once_with(decision)
        self.assertEqual(decision.context[0].content, "Linus Torvalds criou o Linux.")
        self.assertEqual(session.method_calls, [])
        transcribe_audio.assert_not_called()
        speak_response.assert_not_called()
        realtime_audio.assert_not_called()

    def test_rejects_empty_text_or_device_before_classification(self) -> None:
        interaction_service = Mock()
        with patch.object(app, "interaction_service", interaction_service):
            client = TestClient(app.app)
            empty_text = client.post("/interact", json={"text": "   ", "device_id": "client"})
            empty_device = client.post("/interact", json={"text": "oi", "device_id": "   "})

        self.assertEqual(empty_text.status_code, 422)
        self.assertEqual(empty_device.status_code, 422)
        interaction_service.classify.assert_not_called()

    def test_has_no_audio_or_command_session_dependencies(self) -> None:
        result = InteractionResult(text="OlÃ¡.", interaction_state="idle")
        decision = CoreDecision(kind="llm", text="OlÃ¡", device_id="text-client")
        response, _service, session, transcribe_audio, speak_response, realtime_audio = self.post_with_service(decision, result)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(session.method_calls, [])
        transcribe_audio.assert_not_called()
        speak_response.assert_not_called()
        realtime_audio.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
