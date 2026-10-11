"""Testes offline do provider ChatGPT Plan, sem OAuth ou Responses reais."""

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services.auth import AuthError
from services.llm import ChatGPTPlanLLMClient, ConversationMessage, LLMError
from services.llm.chatgpt_plan import DEFAULT_MODEL, OPENAI_BASE_URL, PLAN_ERROR_CODES
from services.llm.personality import ROCKY_SYSTEM_PROMPT


class FakeStream:
    def __init__(self, events):
        self._events = tuple(events)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self._events)


class FakeClient:
    def __init__(self, events):
        self.responses = SimpleNamespace(create=Mock(return_value=FakeStream(events)))


class ClientFactory:
    def __init__(self, events):
        self.events = events
        self.calls = []
        self.client = FakeClient(events)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.client


class SessionDouble:
    def __init__(self, token="oauth-access-token"):
        self.token = token
        self.calls = 0

    def access_token_for_inference(self):
        self.calls += 1
        return self.token


class ChatGPTPlanLLMClientTest(unittest.TestCase):
    def setUp(self) -> None:
        self._verbose = patch.dict(os.environ, {"KODA_VERBOSE": "1"})
        self._verbose.start()

    def tearDown(self) -> None:
        self._verbose.stop()

    @staticmethod
    def completed_events():
        return (
            SimpleNamespace(type="response.output_text.delta", delta="Ola, "),
            SimpleNamespace(type="response.output_text.delta", delta="Koda."),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=11,
                        output_tokens=7,
                        total_tokens=18,
                        input_tokens_details=SimpleNamespace(cached_tokens=2),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
                    )
                ),
            ),
        )

    def build_client(self, events=None, session=None):
        factory = ClientFactory(self.completed_events() if events is None else events)
        return ChatGPTPlanLLMClient(session=session or SessionDouble(), client_factory=factory), factory

    def test_default_provider_uses_oauth_sdk_contract_and_supported_payload(self):
        session = SessionDouble()
        client, factory = self.build_client(session=session)
        context = (ConversationMessage(role="user", content="Pergunta anterior"),
                   ConversationMessage(role="assistant", content="Resposta anterior"))

        with patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-be-used"}):
            self.assertEqual(tuple(client.stream_response("Pergunta atual", context)), ("Ola, ", "Koda."))

        self.assertEqual(session.calls, 1)
        self.assertEqual(factory.calls, [{
            "api_key": "oauth-access-token",
            "base_url": OPENAI_BASE_URL,
            "max_retries": 0,
        }])
        request = factory.client.responses.create.call_args.kwargs
        self.assertEqual(set(request), {"model", "instructions", "input", "store", "stream"})
        self.assertEqual(request["model"], DEFAULT_MODEL)
        self.assertEqual(request["instructions"], ROCKY_SYSTEM_PROMPT)
        self.assertEqual(request["input"], [
            {"role": "user", "content": "Pergunta anterior"},
            {"role": "assistant", "content": "Resposta anterior"},
            {"role": "user", "content": "Pergunta atual"},
        ])
        self.assertFalse(request["store"])
        self.assertTrue(request["stream"])

    def test_ask_makes_one_responses_request_and_returns_the_completed_text(self):
        client, factory = self.build_client()

        result = client.ask("Pergunta")

        self.assertEqual(result.text, "Ola, Koda.")
        self.assertEqual(result.model, DEFAULT_MODEL)
        factory.client.responses.create.assert_called_once()

    def test_completed_usage_is_logged_without_response_content(self):
        client, _ = self.build_client()
        output = io.StringIO()

        with redirect_stdout(output):
            tuple(client.stream_response("Pergunta"))

        self.assertIn("[USAGE] model=gpt-5.6-terra input=11 output=7 total=18 cached=2 reasoning=3", output.getvalue())
        self.assertNotIn("oauth-access-token", output.getvalue())

    def test_failed_event_preserves_supported_plan_error_code(self):
        code = "subscription_sharing_usage_limit_exceeded"
        events = (SimpleNamespace(type="response.failed", response=SimpleNamespace(error=SimpleNamespace(code=code))),)
        client, _ = self.build_client(events)
        output = io.StringIO()

        with redirect_stdout(output), self.assertRaisesRegex(LLMError, code):
            tuple(client.stream_response("Pergunta"))

        self.assertIn(f"error_code={code}", output.getvalue())

    def test_incomplete_and_interrupted_streams_never_succeed(self):
        incomplete = (SimpleNamespace(type="response.incomplete", response=SimpleNamespace(error=None)),)
        for events, expected in ((incomplete, "response_incomplete"), ((), "before response.completed")):
            with self.subTest(expected=expected):
                client, _ = self.build_client(events)
                with self.assertRaisesRegex(LLMError, expected):
                    tuple(client.stream_response("Pergunta"))

    def test_auth_required_does_not_create_client_or_open_login(self):
        session = Mock()
        session.access_token_for_inference.side_effect = AuthError("authentication_required")
        client, factory = self.build_client(session=session)

        with self.assertRaisesRegex(LLMError, "authentication is required"):
            tuple(client.stream_response("Pergunta"))

        factory.client.responses.create.assert_not_called()
        session.login.assert_not_called()

    def test_pending_rotation_is_logged_without_creating_a_client_or_exposing_a_token(self):
        session = Mock()
        session.access_token_for_inference.side_effect = AuthError("refresh_rotation_pending")
        client, factory = self.build_client(session=session)
        output = io.StringIO()

        with redirect_stdout(output), self.assertRaisesRegex(LLMError, "authentication is required"):
            tuple(client.stream_response("Pergunta"))

        self.assertIn("auth_error=refresh_rotation_pending", output.getvalue())
        self.assertNotIn("oauth-access-token", output.getvalue())
        factory.client.responses.create.assert_not_called()

    def test_sdk_failure_is_sanitized_and_never_falls_back_to_openrouter(self):
        client, factory = self.build_client()
        factory.client.responses.create.side_effect = RuntimeError("oauth-access-token SECRET")

        with patch("services.llm.openrouter.urlopen", side_effect=AssertionError("fallback")) as openrouter:
            with self.assertRaisesRegex(LLMError, "ChatGPT Plan request failed"):
                tuple(client.stream_response("Pergunta"))

        openrouter.assert_not_called()

    def test_plan_error_codes_are_explicitly_allowlisted(self):
        self.assertIn("chatpass_v2_scope_not_authorized", PLAN_ERROR_CODES)
        self.assertIn("subscription_sharing_route_not_supported", PLAN_ERROR_CODES)


class ChatGPTPlanDefaultIntegrationTest(unittest.TestCase):
    def test_app_uses_chatgpt_plan_as_default_provider(self):
        import app

        self.assertIsInstance(app.llm_client, ChatGPTPlanLLMClient)


if __name__ == "__main__":
    unittest.main()
