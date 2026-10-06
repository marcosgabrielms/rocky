"""Testes determinísticos da proteção de custo e streaming do OpenRouter."""

import json
import io
import os
import unittest
from contextlib import redirect_stdout
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from services.llm import ConversationMessage, LLMError, OpenRouterLLMClient
from services.latency import latency_scope
from services.llm.openrouter import ALLOWED_FREE_MODELS, DEFAULT_MODEL


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
    def test_default_and_allowlist_are_exact_without_dynamic_routers(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "google/gemma-4-31b-it:free")
        self.assertEqual(ALLOWED_FREE_MODELS, frozenset({
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "thinkingmachines/inkling-small:free",
        }))
        with patch.dict(os.environ, {}, clear=True):
            payload = json.loads(OpenRouterLLMClient(api_key="test-key")._build_request("Pergunta", (), False).data)
        self.assertEqual(payload["model"], DEFAULT_MODEL)

    def test_manual_environment_model_is_used_without_automatic_selection(self) -> None:
        for model in ALLOWED_FREE_MODELS:
            with self.subTest(model=model), patch.dict(os.environ, {"OPENROUTER_MODEL": model}):
                client = OpenRouterLLMClient(api_key="test-key")
                client._validate_free_configuration()
                self.assertEqual(json.loads(client._build_request("Pergunta", (), False).data)["model"], model)

    def test_unapproved_models_make_zero_requests_in_both_paths(self) -> None:
        for model in ("openrouter/free", "openrouter/auto", "provider/paid", "provider/unknown:free",
                      "openai/gpt-oss-20b:free", "", DEFAULT_MODEL + " "):
            with self.subTest(model=model), patch("services.llm.openrouter.urlopen") as request:
                client = OpenRouterLLMClient(api_key="test-key", model=model)
                with self.assertRaises(LLMError):
                    client.ask("Pergunta")
                with self.assertRaises(LLMError):
                    tuple(client.stream_response("Pergunta"))
                request.assert_not_called()

    def test_each_request_locks_price_and_privacy_with_only_one_model(self) -> None:
        for model in ALLOWED_FREE_MODELS:
            for stream in (False, True):
                with self.subTest(model=model, stream=stream):
                    payload = json.loads(OpenRouterLLMClient(api_key="test-key", model=model)._build_request("Pergunta", (), stream).data)
                    self.assertEqual(payload["model"], model)
                    self.assertNotIn("models", payload)
                    self.assertNotIn("route", payload)
                    self.assertEqual(payload["provider"], {
                        "max_price": {"prompt": 0, "completion": 0},
                        "data_collection": "deny", "allow_fallbacks": True,
                    })
                    self.assertEqual(payload["temperature"], 0.2)
                    self.assertEqual(payload["max_tokens"], 120)
                    self.assertTrue(payload["reasoning"]["exclude"])
                    if model == DEFAULT_MODEL:
                        self.assertFalse(payload["reasoning"]["enabled"])
                    else:
                        self.assertNotIn("enabled", payload["reasoning"])
                        self.assertNotIn("effort", payload["reasoning"])

    def test_sync_errors_never_retry_or_switch_model(self) -> None:
        for model in ALLOWED_FREE_MODELS:
            for error in (HTTPError("https://example.test", 404, "unavailable", None, None),
                          TimeoutError(), URLError("offline")):
                with self.subTest(model=model, error=type(error).__name__):
                    with patch("services.llm.openrouter.urlopen", side_effect=error) as request:
                        with self.assertRaises(LLMError):
                            OpenRouterLLMClient(api_key="test-key", model=model).ask("Pergunta")
                    request.assert_called_once()

    def test_reasoning_fields_never_replace_final_content(self) -> None:
        body = json.dumps({"choices": [{"message": {
            "content": "Resposta final.", "reasoning": "NAO FALAR",
            "reasoning_details": [{"text": "SEGREDO INTERNO"}],
        }}]}).encode()
        self.assertEqual(OpenRouterLLMClient._parse_response(body).text, "Resposta final.")
        events = (
            {"choices": [{"delta": {"reasoning": "NAO FALAR", "reasoning_details": [{"text": "SEGREDO"}]}}]},
            {"choices": [{"delta": {"content": "Resposta final.", "reasoning": "NAO FALAR"}}]},
        )
        fragments = tuple(fragment for event in events
                          for fragment in OpenRouterLLMClient._parse_stream_event(json.dumps(event)))
        self.assertEqual(fragments, ("Resposta final.",))

    def test_voice_prompt_is_shared_and_token_limit_is_not_increased(self) -> None:
        request = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)._build_request("Pergunta", (), True)
        payload = json.loads(request.data)
        prompt = payload["messages"][0]["content"]
        for instruction in ("falada em voz alta", "única frase curta", "no máximo duas frases curtas",
                            "Não repita a pergunta", "Não use Markdown", "metatexto", "Input:"):
            self.assertIn(instruction, prompt)
        self.assertEqual(payload["max_tokens"], 120)

    def test_routed_model_is_logged_without_latency_scope_and_without_extra_request(self) -> None:
        event = b'data: {"model":"provider/model:free","choices":[{"delta":{"content":"Resposta"}}]}\n'
        output = io.StringIO()
        with (
            patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=(event, event, b"data: [DONE]\n"))) as request,
            redirect_stdout(output),
        ):
            tuple(OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL).stream_response("Pergunta"))
        request.assert_called_once()
        self.assertEqual(output.getvalue().count("[LLM] routed_model=provider/model:free"), 1)
    def test_only_explicitly_allowed_free_models_are_accepted(self) -> None:
        for model in ALLOWED_FREE_MODELS:
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
        client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
        context = (ConversationMessage(role="user", content="Contexto"),)
        ask_request = client._build_request("Pergunta", context, stream=False)
        stream_request = client._build_request("Pergunta", context, stream=True)
        ask_payload = json.loads(ask_request.data)
        stream_payload = json.loads(stream_request.data)

        self.assertEqual(ask_request.full_url, stream_request.full_url)
        self.assertEqual(ask_request.get_method(), "POST")
        self.assertEqual(stream_request.get_method(), "POST")
        self.assertEqual(ask_payload["model"], DEFAULT_MODEL)
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
                client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
                with patch("services.llm.openrouter.urlopen", side_effect=error) as urlopen_mock:
                    with self.assertRaises(LLMError):
                        tuple(client.stream_response("Pergunta"))

                urlopen_mock.assert_called_once()

    def test_ask_sends_one_non_stream_request(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
        response_body = b'{"choices":[{"message":{"content":"Resposta"}}]}'
        with patch("services.llm.openrouter.urlopen", return_value=FakeResponse(response_body)) as urlopen_mock:
            result = client.ask("Pergunta")

        request = urlopen_mock.call_args.args[0]
        self.assertEqual(result.text, "Resposta")
        self.assertFalse(json.loads(request.data)["stream"])

    def test_stream_sends_one_stream_request(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
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

    def test_routed_model_is_read_from_existing_stream_without_extra_request(self) -> None:
        client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
        event = b'data: {"model":"provider/example:free","choices":[{"delta":{"content":"Resposta"}}]}\n'
        output = io.StringIO()
        with (
            patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=(event, event, b"data: [DONE]\n"))) as request,
            redirect_stdout(output),
            latency_scope(),
        ):
            self.assertEqual(tuple(client.stream_response("Pergunta")), ("Resposta", "Resposta"))
        request.assert_called_once()
        self.assertEqual(output.getvalue().count("[LLM] routed_model=provider/example:free"), 1)


if __name__ == "__main__":
    unittest.main()
