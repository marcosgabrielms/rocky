"""Testes offline do provider Gemini, da reserva e da escolha de provider."""

import io
import os
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from services.llm import (
    ChatGPTPlanLLMClient,
    ConversationMessage,
    FallbackLLMClient,
    GeminiLLMClient,
    LLMError,
    StreamingLLMClient,
    build_llm_client,
)
from services.llm.factory import (
    DEFAULT_GEMINI_FALLBACK_MODELS,
    SLOW_MODEL_TIMEOUT_MS,
    gemini_models,
    selected_provider,
)
from services.llm.gemini import DEFAULT_MODEL, REQUEST_TIMEOUT_MS
from services.llm.personality import ROCKY_SYSTEM_PROMPT


def _chunk(*texts, thought=False, usage=None):
    parts = [SimpleNamespace(text=text, thought=thought) for text in texts]
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"api error {code} with private body")
        self.code = code


class ClientFactory:
    def __init__(self, chunks=(), error=None):
        self.calls = []
        self.generate = Mock(side_effect=self._generate)
        self._chunks = tuple(chunks)
        self._error = error
        self.client = SimpleNamespace(models=SimpleNamespace(generate_content_stream=self.generate))

    def _generate(self, **_kwargs):
        if self._error is not None:
            raise self._error
        return iter(self._chunks)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.client


class ScriptedClient:
    def __init__(self, chunks=(), error=None, fail_after=None):
        self.chunks = tuple(chunks)
        self.error = error
        self.fail_after = fail_after
        self.calls = 0

    def ask(self, message, context=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text="".join(self.chunks))

    def stream_response(self, message, context=None):
        self.calls += 1
        for index, chunk in enumerate(self.chunks):
            if self.fail_after is not None and index == self.fail_after:
                raise LLMError("broke mid-stream")
            yield chunk
        if self.error is not None:
            raise self.error


class GeminiLLMClientTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {"KODA_VERBOSE": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _stream(self, client, *args, **kwargs):
        with redirect_stdout(io.StringIO()) as output:
            chunks = list(client.stream_response(*args, **kwargs))
        return chunks, output.getvalue()

    def test_streams_text_with_personality_and_timeout(self) -> None:
        factory = ClientFactory([_chunk("Oi, "), _chunk("humano.")])
        client = GeminiLLMClient(api_key="key", client_factory=factory)

        chunks, _ = self._stream(client, "Olá")

        self.assertEqual(chunks, ["Oi, ", "humano."])
        self.assertEqual(factory.calls, [{"api_key": "key", "http_options": {"timeout": REQUEST_TIMEOUT_MS}}])
        kwargs = factory.generate.call_args.kwargs
        self.assertEqual(kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(kwargs["config"], {"system_instruction": ROCKY_SYSTEM_PROMPT})
        self.assertEqual(kwargs["contents"], [{"role": "user", "parts": [{"text": "Olá"}]}])

    def test_maps_roles_and_moves_system_context_into_instruction(self) -> None:
        factory = ClientFactory([_chunk("ok")])
        client = GeminiLLMClient(api_key="key", client_factory=factory)
        context = (
            ConversationMessage(role="system", content="Memória: gosta de café."),
            ConversationMessage(role="user", content="oi"),
            ConversationMessage(role="assistant", content="fala"),
        )

        self._stream(client, "e aí?", context)

        kwargs = factory.generate.call_args.kwargs
        self.assertEqual(
            kwargs["config"]["system_instruction"], ROCKY_SYSTEM_PROMPT + "\n\nMemória: gosta de café."
        )
        self.assertEqual(
            [item["role"] for item in kwargs["contents"]], ["user", "model", "user"]
        )

    def test_skips_thought_parts_and_empty_chunks(self) -> None:
        factory = ClientFactory([_chunk("pensando...", thought=True), SimpleNamespace(candidates=None), _chunk("Resposta.")])
        client = GeminiLLMClient(api_key="key", client_factory=factory)

        chunks, _ = self._stream(client, "pergunta")

        self.assertEqual(chunks, ["Resposta."])

    def test_ask_joins_stream_and_reports_model(self) -> None:
        factory = ClientFactory([_chunk(" Duas "), _chunk("partes. ")])
        client = GeminiLLMClient(api_key="key", model="gemma-test", client_factory=factory)

        with redirect_stdout(io.StringIO()):
            result = client.ask("oi")

        self.assertEqual((result.text, result.model), ("Duas partes.", "gemma-test"))

    def test_empty_response_is_an_error(self) -> None:
        client = GeminiLLMClient(api_key="key", client_factory=ClientFactory([]))

        with redirect_stdout(io.StringIO()), self.assertRaises(LLMError):
            client.ask("oi")

    def test_missing_key_fails_without_creating_client(self) -> None:
        factory = ClientFactory([_chunk("nunca")])
        client = GeminiLLMClient(client_factory=factory)

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), self.assertRaises(LLMError):
            self._stream(client, "oi")

        self.assertEqual(factory.calls, [])

    def test_reads_key_from_environment_once(self) -> None:
        factory = ClientFactory([_chunk("ok")])
        client = GeminiLLMClient(client_factory=factory)

        with patch.dict(os.environ, {"GEMINI_API_KEY": " env-key "}):
            self._stream(client, "um")
            self._stream(client, "dois")

        self.assertEqual([call["api_key"] for call in factory.calls], ["env-key"])

    def test_quota_error_is_reported_without_leaking_body(self) -> None:
        client = GeminiLLMClient(api_key="key", client_factory=ClientFactory(error=FakeAPIError(429)))

        with redirect_stdout(io.StringIO()) as output, self.assertRaises(LLMError) as raised:
            list(client.stream_response("oi"))

        self.assertIn("quota_exhausted", str(raised.exception))
        self.assertIn("error_code=quota_exhausted", output.getvalue())
        self.assertNotIn("private body", output.getvalue())

    def test_unknown_error_uses_generic_code(self) -> None:
        client = GeminiLLMClient(api_key="key", client_factory=ClientFactory(error=RuntimeError("secret")))

        with redirect_stdout(io.StringIO()) as output, self.assertRaises(LLMError):
            list(client.stream_response("oi"))

        self.assertIn("error_code=request_failed", output.getvalue())
        self.assertNotIn("secret", output.getvalue())

    def test_logs_usage_from_last_chunk(self) -> None:
        usage = SimpleNamespace(prompt_token_count=120, candidates_token_count=15, total_token_count=135)
        client = GeminiLLMClient(api_key="key", client_factory=ClientFactory([_chunk("ok", usage=usage)]))

        _, output = self._stream(client, "oi")

        self.assertIn(f"[USAGE] model={DEFAULT_MODEL} input=120 output=15 total=135", output)

    def test_is_a_streaming_client(self) -> None:
        self.assertIsInstance(GeminiLLMClient(api_key="key"), StreamingLLMClient)


class FallbackLLMClientTest(unittest.TestCase):
    def _stream(self, client):
        with redirect_stdout(io.StringIO()):
            return list(client.stream_response("oi"))

    def test_uses_primary_when_it_works(self) -> None:
        primary, backup = ScriptedClient(["a", "b"]), ScriptedClient(["x"])

        self.assertEqual(self._stream(FallbackLLMClient((primary, backup))), ["a", "b"])
        self.assertEqual(backup.calls, 0)

    def test_switches_when_primary_fails_before_text(self) -> None:
        primary, backup = ScriptedClient(error=LLMError("quota")), ScriptedClient(["reserva"])

        self.assertEqual(self._stream(FallbackLLMClient((primary, backup))), ["reserva"])

    def test_never_mixes_answers_after_primary_started(self) -> None:
        primary = ScriptedClient(["começo ", "fim"], fail_after=1)
        backup = ScriptedClient(["outra resposta"])
        received = []

        with redirect_stdout(io.StringIO()), self.assertRaises(LLMError):
            for chunk in FallbackLLMClient((primary, backup)).stream_response("oi"):
                received.append(chunk)

        self.assertEqual(received, ["começo "])
        self.assertEqual(backup.calls, 0)

    def test_raises_last_error_when_all_fail(self) -> None:
        clients = (ScriptedClient(error=LLMError("first")), ScriptedClient(error=LLMError("last")))

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(LLMError, "last"):
            list(FallbackLLMClient(clients).stream_response("oi"))

    def test_ask_falls_back(self) -> None:
        clients = (ScriptedClient(error=LLMError("down")), ScriptedClient(["ok"]))

        with redirect_stdout(io.StringIO()):
            self.assertEqual(FallbackLLMClient(clients).ask("oi").text, "ok")

    def test_requires_clients(self) -> None:
        with self.assertRaises(ValueError):
            FallbackLLMClient(())


class BuildLLMClientTest(unittest.TestCase):
    def test_defaults_to_lite_chain_ending_in_gemma(self) -> None:
        client = build_llm_client({})

        self.assertIsInstance(client, FallbackLLMClient)
        models = [inner.model for inner in client._clients]
        self.assertEqual(models, [DEFAULT_MODEL, *DEFAULT_GEMINI_FALLBACK_MODELS])
        self.assertEqual(models[-1], "gemma-4-31b-it")

    def test_only_slow_gemma_gets_long_timeout(self) -> None:
        client = build_llm_client({})

        timeouts = {inner.model: inner._timeout_ms for inner in client._clients}
        self.assertEqual(timeouts["gemma-4-31b-it"], SLOW_MODEL_TIMEOUT_MS)
        self.assertEqual(timeouts[DEFAULT_MODEL], REQUEST_TIMEOUT_MS)

    def test_custom_models_and_disabled_fallback(self) -> None:
        client = build_llm_client({"KODA_GEMINI_MODEL": "m1", "KODA_GEMINI_FALLBACK_MODELS": ""})

        self.assertIsInstance(client, GeminiLLMClient)
        self.assertEqual(client.model, "m1")

    def test_fallback_list_is_trimmed_and_deduplicated(self) -> None:
        models = gemini_models({"KODA_GEMINI_MODEL": "m1", "KODA_GEMINI_FALLBACK_MODELS": " m2 , m1,,m2 "})

        self.assertEqual(models, ("m1", "m2"))

    def test_chatgpt_plan_remains_selectable(self) -> None:
        self.assertIsInstance(build_llm_client({"KODA_LLM_PROVIDER": " ChatGPT_Plan "}), ChatGPTPlanLLMClient)

    def test_rejects_unknown_provider(self) -> None:
        with self.assertRaises(ValueError):
            selected_provider({"KODA_LLM_PROVIDER": "openrouter"})


class LauncherProviderTest(unittest.TestCase):
    def test_launcher_requires_gemini_key_and_keeps_chatgpt_auth_optional(self) -> None:
        from pathlib import Path

        script = (Path(__file__).parent.parent / "tools/rocky_launcher.ps1").read_text(encoding="utf-8")
        self.assertIn("KODA_LLM_PROVIDER", script)
        self.assertIn("GEMINI_API_KEY", script)
        self.assertLess(script.index('-eq "gemini"'), script.index("chatgpt_auth.py"))
        self.assertNotRegex(script, r"GEMINI_API_KEY\s*=\s*\S")


if __name__ == "__main__":
    unittest.main()
