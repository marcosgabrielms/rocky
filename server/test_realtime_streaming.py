"""Testes determinísticos da proteção de custo e streaming do OpenRouter."""

import json
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from services.llm import ConversationMessage, LLMError, OpenRouterLLMClient


class FakeResponse:
    def __init__(self, body: bytes = b"", lines: tuple[bytes, ...] = ()) -> None:
        self._body = body
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._lines)


class OpenRouterStreamingTest(unittest.TestCase):
    def test_only_explicitly_allowed_free_models_are_accepted(self) -> None:
        for model in ("openrouter/free", "openai/gpt-oss-20b:free"):
            with self.subTest(model=model):
                OpenRouterLLMClient(api_key="test-key", model=model)._validate_free_configuration()

    def test_paid_model_is_blocked_before_any_http_request(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model="provider/paid-model")

        with patch("services.llm.openrouter.urlopen") as urlopen_mock:
            with self.assertRaisesRegex(LLMError, "not explicitly free"):
                client.ask("Pergunta")

            with self.assertRaisesRegex(LLMError, "not explicitly free"):
                tuple(client.stream_response("Pergunta"))

        urlopen_mock.assert_not_called()

    def test_free_suffix_is_not_an_implicit_allowlist(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model="provider/unknown:free")

        with patch("services.llm.openrouter.urlopen") as urlopen_mock:
            with self.assertRaisesRegex(LLMError, "not explicitly free"):
                client.ask("Pergunta")

        urlopen_mock.assert_not_called()

    def test_ask_and_stream_share_endpoint_model_and_messages(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model="openrouter/free")
        context = (ConversationMessage(role="user", content="Contexto"),)
        ask_request = client._build_request("Pergunta", context, stream=False)
        stream_request = client._build_request("Pergunta", context, stream=True)
        ask_payload = json.loads(ask_request.data)
        stream_payload = json.loads(stream_request.data)

        self.assertEqual(ask_request.full_url, stream_request.full_url)
        self.assertEqual(ask_request.get_method(), "POST")
        self.assertEqual(stream_request.get_method(), "POST")
        self.assertEqual(ask_payload["model"], "openrouter/free")
        self.assertEqual(ask_payload["messages"], stream_payload["messages"])
        self.assertFalse(ask_payload["stream"])
        self.assertTrue(stream_payload["stream"])
        self.assertEqual(ask_request.get_header("Accept"), "application/json")
        self.assertEqual(stream_request.get_header("Accept"), "text/event-stream")

    def test_stream_errors_before_tokens_make_exactly_one_request(self) -> None:
        for error in (
            HTTPError("https://example.test", 404, "not found", None, None),
            HTTPError("https://example.test", 402, "payment", None, None),
            HTTPError("https://example.test", 429, "rate", None, None),
            HTTPError("https://example.test", 500, "server", None, None),
            TimeoutError(),
            URLError("offline"),
        ):
            with self.subTest(error=type(error).__name__):
                client = OpenRouterLLMClient(api_key="test-key", model="openrouter/free")
                with patch("services.llm.openrouter.urlopen", side_effect=error) as urlopen_mock:
                    with self.assertRaises(LLMError):
                        tuple(client.stream_response("Pergunta"))

                urlopen_mock.assert_called_once()

    def test_ask_sends_one_non_stream_request(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model="openrouter/free")
        response_body = b'{"choices":[{"message":{"content":"Resposta"}}]}'
        with patch("services.llm.openrouter.urlopen", return_value=FakeResponse(response_body)) as urlopen_mock:
            result = client.ask("Pergunta")

        request = urlopen_mock.call_args.args[0]
        self.assertEqual(result.text, "Resposta")
        self.assertFalse(json.loads(request.data)["stream"])

    def test_stream_sends_one_stream_request(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model="openrouter/free")
        lines = (b'data: {"choices":[{"delta":{"content":"Resposta"}}]}\n', b"data: [DONE]\n")
        with patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=lines)) as urlopen_mock:
            result = tuple(client.stream_response("Pergunta"))

        request = urlopen_mock.call_args.args[0]
        self.assertEqual(result, ("Resposta",))
        self.assertTrue(json.loads(request.data)["stream"])

    def test_stream_event_produces_only_delta_content(self) -> None:
        event = '{"choices":[{"delta":{"content":"Olá"}}]}'

        self.assertEqual(tuple(OpenRouterLLMClient._parse_stream_event(event)), ("Olá",))

    def test_stream_event_with_provider_error_is_controlled(self) -> None:
        event = '{"error":{"message":"rate limited"}}'

        with self.assertRaises(LLMError):
            tuple(OpenRouterLLMClient._parse_stream_event(event))


if __name__ == "__main__":
    unittest.main()
