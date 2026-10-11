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
from services.llm.openrouter import (
    ALLOWED_FREE_MODELS, DEFAULT_MODEL, GEMMA_MODEL,
    MAX_ERROR_BODY_BYTES, STREAM_DEADLINE_SECONDS, _StreamDeadline,
)


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
    def setUp(self) -> None:
        self._verbose = patch.dict(os.environ, {"KODA_VERBOSE": "1"})
        self._verbose.start()

    def tearDown(self) -> None:
        self._verbose.stop()

    def test_total_deadline_covers_keepalives_without_retry(self) -> None:
        from types import SimpleNamespace
        clock = SimpleNamespace(now=0.0)

        class KeepaliveResponse(FakeResponse):
            def __iter__(self):
                while True:
                    clock.now += 61
                    yield b": keepalive\n"

        with (patch("services.llm.openrouter.time.monotonic", side_effect=lambda: clock.now),
              patch("services.llm.openrouter.Timer") as timer,
              patch("services.llm.openrouter.urlopen", return_value=KeepaliveResponse()) as request):
            client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
            with self.assertRaisesRegex(LLMError, "total deadline"):
                tuple(client.stream_response("Pergunta"))
        request.assert_called_once()
        timer.assert_called_once()
        self.assertEqual(timer.call_args.args[0], STREAM_DEADLINE_SECONDS)
        timer.return_value.cancel.assert_called_once()

    def test_watchdog_unblocks_socket_without_waiting_for_a_complete_sse_line(self) -> None:
        import socket
        from types import SimpleNamespace
        from unittest.mock import Mock
        connection = Mock()
        response = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=connection)))
        with (patch("services.llm.openrouter.time.monotonic", return_value=0),
              patch("services.llm.openrouter.Timer") as timer):
            with _StreamDeadline(response, STREAM_DEADLINE_SECONDS):
                timer.call_args.args[1]()
        connection.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        timer.return_value.start.assert_called_once()
        timer.return_value.cancel.assert_called_once()

    def test_network_errors_become_safe_llm_error_in_both_paths(self) -> None:
        from http.client import IncompleteRead
        from ssl import SSLError
        for error in (ConnectionResetError("SECRET"), IncompleteRead(b"SECRET"), SSLError("SECRET")):
            for stream in (False, True):
                with self.subTest(error=type(error).__name__, stream=stream):
                    with patch("services.llm.openrouter.urlopen", side_effect=error) as request:
                        client = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL)
                        with self.assertRaises(LLMError) as caught:
                            if stream:
                                tuple(client.stream_response("Pergunta"))
                            else:
                                client.ask("Pergunta")
                    request.assert_called_once()
                    self.assertNotIn("SECRET", str(caught.exception))

    def test_read_failure_after_first_fragment_is_safe_and_not_retried(self) -> None:
        from http.client import IncompleteRead

        class InterruptedResponse(FakeResponse):
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"content":"Resposta."}}]}\n'
                raise IncompleteRead(b"SECRET")

        with patch("services.llm.openrouter.urlopen", return_value=InterruptedResponse()) as request:
            stream = OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL).stream_response("Pergunta")
            self.assertEqual(next(stream), "Resposta.")
            with self.assertRaises(LLMError) as caught:
                next(stream)
        request.assert_called_once()
        self.assertNotIn("SECRET", str(caught.exception))

    def test_default_and_allowlist_are_exact_without_dynamic_routers(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "inclusionai/ling-3.1-flash")
        self.assertEqual(ALLOWED_FREE_MODELS, frozenset({
            "google/gemma-4-31b-it:free",
            "inclusionai/ling-3.1-flash",
        }))
        with patch.dict(os.environ, {}, clear=True):
            payload = json.loads(OpenRouterLLMClient(api_key="test-key")._build_request("Pergunta", (), False).data)
        self.assertEqual(payload["model"], "inclusionai/ling-3.1-flash")
        self.assertNotIn("models", payload)

    def test_manual_environment_model_is_used_without_automatic_selection(self) -> None:
        for model in ALLOWED_FREE_MODELS:
            for stream in (False, True):
                with self.subTest(model=model, stream=stream), patch.dict(os.environ, {"OPENROUTER_MODEL": model}):
                    response = FakeResponse(
                        body=b'{"choices":[{"message":{"content":"Resposta."}}]}',
                        lines=(b'data: {"choices":[{"delta":{"content":"Resposta."}}]}\n', b'data: [DONE]\n'),
                    )
                    with patch("services.llm.openrouter.urlopen", return_value=response) as request:
                        client = OpenRouterLLMClient(api_key="test-key")
                        result = tuple(client.stream_response("Pergunta")) if stream else client.ask("Pergunta").text
                    request.assert_called_once()
                    payload = json.loads(request.call_args.args[0].data)
                    self.assertEqual(payload["model"], model)
                    self.assertNotIn("models", payload)
                    self.assertEqual(result, ("Resposta.",) if stream else "Resposta.")

    def test_removed_environment_overrides_are_blocked_before_http(self) -> None:
        for model in ("nvidia/nemotron-3-super-120b-a12b:free", "thinkingmachines/inkling-small:free"):
            with self.subTest(model=model), patch.dict(os.environ, {"OPENROUTER_MODEL": model}):
                with patch("services.llm.openrouter.urlopen") as request:
                    client = OpenRouterLLMClient(api_key="test-key")
                    with self.assertRaises(LLMError):
                        client.ask("Pergunta")
                    with self.assertRaises(LLMError):
                        tuple(client.stream_response("Pergunta"))
                    request.assert_not_called()

    def test_default_accepts_ling_sync_response_without_second_request(self) -> None:
        body = json.dumps({
            "model": "inclusionai/ling-3.1-flash",
            "choices": [{"message": {"content": "Resposta final.", "reasoning": "NEVER SPEAK"}}],
        }).encode()
        with patch.dict(os.environ, {"KODA_VERBOSE": "1"}, clear=True), redirect_stdout(io.StringIO()) as logs:
            with patch("services.llm.openrouter.urlopen", return_value=FakeResponse(body)) as request:
                result = OpenRouterLLMClient(api_key="test-key").ask("Pergunta")
        request.assert_called_once()
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["model"], "inclusionai/ling-3.1-flash")
        self.assertNotIn("models", payload)
        self.assertEqual(result.text, "Resposta final.")
        self.assertIn("[LLM] routed_model=inclusionai/ling-3.1-flash", logs.getvalue())
        self.assertNotIn("NEVER SPEAK", logs.getvalue())

    def test_unapproved_models_make_zero_requests_in_both_paths(self) -> None:
        for model in ("openrouter/free", "openrouter/auto", "provider/paid", "provider/unknown:free",
                      "nvidia/nemotron-3-super-120b-a12b:free", "thinkingmachines/inkling-small:free",
                      "inclusionai/ling-3.1-flash:free",
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
                    if model == GEMMA_MODEL:
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
        event = b'data: {"model":"inclusionai/ling-3.1-flash","choices":[{"delta":{"content":"Resposta"}}]}\n'
        output = io.StringIO()
        with (
            patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=(event, event, b"data: [DONE]\n"))) as request,
            patch.dict(os.environ, {"KODA_VERBOSE": "1"}),
            redirect_stdout(output),
        ):
            tuple(OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL).stream_response("Pergunta"))
        request.assert_called_once()
        self.assertEqual(output.getvalue().count("[LLM] routed_model=inclusionai/ling-3.1-flash"), 1)

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
        event = b'data: {"model":"inclusionai/ling-3.1-flash","choices":[{"delta":{"content":"Resposta"}}]}\n'
        output = io.StringIO()
        with (
            patch("services.llm.openrouter.urlopen", return_value=FakeResponse(lines=(event, event, b"data: [DONE]\n"))) as request,
            patch.dict(os.environ, {"KODA_VERBOSE": "1"}),
            redirect_stdout(output),
            latency_scope(),
        ):
            self.assertEqual(tuple(client.stream_response("Pergunta")), ("Resposta", "Resposta"))
        request.assert_called_once()
        self.assertEqual(output.getvalue().count("[LLM] routed_model=inclusionai/ling-3.1-flash"), 1)

    def test_default_keeps_singular_model_price_privacy_and_reasoning_in_both_paths(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = OpenRouterLLMClient(api_key="test-key")
        for stream in (False, True):
            payload = json.loads(client._build_request("Pergunta", (), stream).data)
            self.assertEqual(payload["model"], "inclusionai/ling-3.1-flash")
            self.assertNotIn("models", payload)
            self.assertNotIn("route", payload)
            self.assertEqual(payload["provider"], {
                "max_price": {"prompt": 0, "completion": 0},
                "data_collection": "deny", "allow_fallbacks": True,
            })
            self.assertEqual(payload["reasoning"], {"exclude": True})
            self.assertEqual(payload["max_tokens"], 120)
            self.assertEqual(payload["temperature"], 0.2)

    def test_default_success_uses_exactly_one_request_in_both_paths(self) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream), patch.dict(os.environ, {}, clear=True):
                response = FakeResponse(
                    body=b'{"choices":[{"message":{"content":"Resposta."}}]}',
                    lines=(b'data: {"choices":[{"delta":{"content":"Resposta."}}]}\n', b'data: [DONE]\n'),
                )
                with patch("services.llm.openrouter.urlopen", return_value=response) as request:
                    client = OpenRouterLLMClient(api_key="test-key")
                    result = tuple(client.stream_response("Pergunta")) if stream else client.ask("Pergunta").text
                request.assert_called_once()
                self.assertEqual(result, ("Resposta.",) if stream else "Resposta.")
                payload = json.loads(request.call_args.args[0].data)
                self.assertEqual(payload["model"], DEFAULT_MODEL)
                self.assertNotIn("models", payload)

    def test_environment_cannot_select_multiple_models_or_dynamic_router(self) -> None:
        for model in (f"{DEFAULT_MODEL},{GEMMA_MODEL}", json.dumps([DEFAULT_MODEL, GEMMA_MODEL]),
                      "", "openrouter/free"):
            with self.subTest(model=model), patch.dict(os.environ, {"OPENROUTER_MODEL": model}):
                with patch("services.llm.openrouter.urlopen") as request:
                    client = OpenRouterLLMClient(api_key="test-key")
                    with self.assertRaises(LLMError):
                        client.ask("Pergunta")
                    with self.assertRaises(LLMError):
                        tuple(client.stream_response("Pergunta"))
                    request.assert_not_called()

    def test_default_final_http_errors_never_retry_or_try_gemma_in_both_paths(self) -> None:
        for status in (402, 404, 429, 500, 503):
            for stream in (False, True):
                with self.subTest(status=status, stream=stream), patch.dict(os.environ, {}, clear=True):
                    error = HTTPError("https://example.test", status, "failure", None, io.BytesIO(b'{}'))
                    with patch("services.llm.openrouter.urlopen", side_effect=error) as request:
                        client = OpenRouterLLMClient(api_key="test-key")
                        with self.assertRaises(LLMError):
                            tuple(client.stream_response("Pergunta")) if stream else client.ask("Pergunta")
                    request.assert_called_once()
                    payload = json.loads(request.call_args.args[0].data)
                    self.assertEqual(payload["model"], "inclusionai/ling-3.1-flash")
                    self.assertNotIn("models", payload)

    def test_http_429_diagnostics_project_only_safe_fields_without_secrets(self) -> None:
        for stream in (False, True):
            with self.subTest(stream=stream), patch.dict(os.environ, {}, clear=True):
                sensitive_values = ("sk-secret-test-key", "PROMPT-PRIVATE", "HISTORY-PRIVATE", "person@example.test")
                private_text = " ".join(sensitive_values)
                body = json.dumps({"error": {
                    "code": 429, "message": "Model rate limit exceeded: " + private_text,
                    "metadata": {"error_type": "rate_limit_exceeded", "provider_code": "rate_limited",
                                 "provider_name": "Chutes", "model": DEFAULT_MODEL,
                                 "raw": private_text, "authorization": private_text, "user_id": private_text},
                }}).encode()
                error = HTTPError("https://example.test", 429, private_text, None, io.BytesIO(body))
                logs = io.StringIO()
                with (patch("services.llm.openrouter.urlopen", side_effect=error) as request,
                      patch.dict(os.environ, {"KODA_VERBOSE": "1"}), redirect_stdout(logs)):
                    client = OpenRouterLLMClient(api_key=sensitive_values[0])
                    context = (ConversationMessage(role="user", content=sensitive_values[2]),)
                    with self.assertRaisesRegex(LLMError, "HTTP 429"):
                        if stream:
                            tuple(client.stream_response(sensitive_values[1], context))
                        else:
                            client.ask(sensitive_values[1], context)
                request.assert_called_once()
                self.assertIn("[LLM] http_status=429", logs.getvalue())
                self.assertIn("[LLM] error_code=429", logs.getvalue())
                self.assertIn('error_message="Rate limit reported; scope=model."', logs.getvalue())
                self.assertIn("metadata.error_type=rate_limit_exceeded", logs.getvalue())
                self.assertIn("metadata.provider_code=rate_limited", logs.getvalue())
                self.assertIn("metadata.provider_name=Chutes", logs.getvalue())
                for value in sensitive_values:
                    self.assertNotIn(value, logs.getvalue())
                self.assertNotIn("metadata.raw", logs.getvalue())

    def test_unknown_remote_values_and_personal_information_are_not_logged(self) -> None:
        secret = "PRIVATE-KEY-PROMPT-HISTORY person@example.test"
        body = json.dumps({"error": {"code": secret, "message": secret,
                          "metadata": {"error_type": secret, "provider_code": secret,
                                       "provider_name": secret, "model": secret}}}).encode()
        error = HTTPError("https://example.test", 429, secret, None, io.BytesIO(body))
        logs = io.StringIO()
        with redirect_stdout(logs):
            OpenRouterLLMClient._log_http_error(error)
            OpenRouterLLMClient._log_routed_model(secret)
        self.assertNotIn(secret, logs.getvalue())
        self.assertNotIn("error_code=", logs.getvalue())
        self.assertNotIn("metadata.", logs.getvalue())
        self.assertIn("details omitted", logs.getvalue())
        self.assertIn("routed_model=[unrecognized]", logs.getvalue())

    def test_unreadable_invalid_and_oversized_error_bodies_do_not_mask_llm_error(self) -> None:
        from unittest.mock import Mock
        for body in (b"SECRET-not-json", b"SECRET" * MAX_ERROR_BODY_BYTES,
                     b"[" * 1200 + b"0" + b"]" * 1200):
            logs = io.StringIO()
            error = HTTPError("https://example.test", 429, "failure", None, io.BytesIO(body))
            with patch("services.llm.openrouter.urlopen", side_effect=error) as request, redirect_stdout(logs):
                with self.assertRaisesRegex(LLMError, "HTTP 429"):
                    OpenRouterLLMClient(api_key="test-key", model=DEFAULT_MODEL).ask("Pergunta")
            request.assert_called_once()
            self.assertIn("http_status=429", logs.getvalue())
            self.assertNotIn("SECRET", logs.getvalue())
        stream = Mock()
        stream.closed = False
        stream.read.side_effect = OSError("SECRET")
        logs = io.StringIO()
        with redirect_stdout(logs):
            OpenRouterLLMClient._log_http_error(HTTPError("https://example.test", 429, "failure", None, stream))
        stream.read.assert_called_once_with(MAX_ERROR_BODY_BYTES + 1)
        self.assertNotIn("SECRET", logs.getvalue())

    def test_sync_routed_model_is_optional_and_logs_only_known_slugs(self) -> None:
        for model in (None, DEFAULT_MODEL, GEMMA_MODEL.removesuffix(":free")):
            with self.subTest(model=model), patch.dict(os.environ, {}, clear=True):
                payload = {"choices": [{"message": {"content": "Resposta."}}]}
                if model is not None:
                    payload["model"] = model
                logs = io.StringIO()
                with (patch("services.llm.openrouter.urlopen", return_value=FakeResponse(json.dumps(payload).encode())) as request,
                      redirect_stdout(logs)):
                    self.assertEqual(OpenRouterLLMClient(api_key="test-key").ask("Pergunta").text, "Resposta.")
                request.assert_called_once()
                self.assertNotIn("routed_model=", logs.getvalue())

            if model is not None:
                verbose_logs = io.StringIO()
                with (patch.dict(os.environ, {"KODA_VERBOSE": "1"}, clear=True),
                      patch("services.llm.openrouter.urlopen", return_value=FakeResponse(json.dumps(payload).encode())) as request,
                      redirect_stdout(verbose_logs)):
                    self.assertEqual(OpenRouterLLMClient(api_key="test-key").ask("Pergunta").text, "Resposta.")
                request.assert_called_once()
                self.assertIn(f"routed_model={model}", verbose_logs.getvalue())

    def test_sse_error_diagnostics_remain_safe_and_controlled(self) -> None:
        logs = io.StringIO()
        event = json.dumps({"error": {"code": "provider_overloaded", "message": "Capacity exhausted SECRET"}})
        with redirect_stdout(logs), self.assertRaises(LLMError):
            tuple(OpenRouterLLMClient._parse_stream_event(event))
        self.assertIn("error_code=provider_overloaded", logs.getvalue())
        self.assertIn("Capacity limit reported.", logs.getvalue())
        self.assertNotIn("SECRET", logs.getvalue())

    def test_json_error_under_http_200_is_controlled_without_echoing_body(self) -> None:
        logs = io.StringIO()
        body = json.dumps({"error": {"code": 429, "message": "Rate limit exceeded SECRET"}}).encode()
        with redirect_stdout(logs), self.assertRaisesRegex(LLMError, "error response"):
            OpenRouterLLMClient._parse_response(body)
        self.assertIn("error_code=429", logs.getvalue())
        self.assertNotIn("SECRET", logs.getvalue())

    def test_rate_limit_scope_is_not_assumed_from_http_status_alone(self) -> None:
        messages = {
            "Rate limit exceeded": "unspecified",
            "Account rate-limit exceeded": "account",
            "Upstream rate_limit exceeded": "upstream",
            "Provider rate limit exceeded": "provider",
        }
        for message, scope in messages.items():
            self.assertEqual(OpenRouterLLMClient._safe_error_message(message), f"Rate limit reported; scope={scope}.")


if __name__ == "__main__":
    unittest.main()
