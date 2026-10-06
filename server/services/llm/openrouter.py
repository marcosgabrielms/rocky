"""Cliente HTTP mínimo para a API Chat Completions do OpenRouter."""

import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.llm.base import ConversationMessage, LLMError, LLMResult
from services.llm.personality import ROCKY_SYSTEM_PROMPT
from services.latency import mark_latency


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-4-31b-it:free"
ALLOWED_FREE_MODELS = frozenset({
    DEFAULT_MODEL,
    "nvidia/nemotron-3-super-120b-a12b:free",
    "thinkingmachines/inkling-small:free",
})
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESPONSE_TOKENS = 120
RESPONSE_TEMPERATURE = 0.2


class OpenRouterLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model if model is not None else os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        self._validate_free_configuration()

        request = self._build_request(message, context, stream=False)
        self._log_request(stream=False)
        started_at = time.monotonic()
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response_body = response.read()
        except HTTPError as error:
            print(f"[LLM] http_status={error.code}")
            raise LLMError(f"OpenRouter HTTP {error.code}") from error
        except TimeoutError as error:
            raise LLMError("OpenRouter request timed out") from error
        except URLError as error:
            raise LLMError("OpenRouter connection failed") from error

        elapsed_ms = round((time.monotonic() - started_at) * 1000)
        result = self._parse_response(response_body)
        print("[LLM] response received")
        print(f"[LLM] elapsed_ms={elapsed_ms}")
        if result.model:
            self._log_routed_model(result.model)
        return result

    def stream_response(
        self, message: str, context: Sequence[ConversationMessage] | None = None
    ) -> Iterator[str]:
        """Produz os fragmentos SSE de uma única geração OpenRouter."""
        self._validate_free_configuration()
        request = self._build_request(message, context, stream=True)
        self._log_request(stream=True)
        model_logged = False

        def log_model(model: str) -> None:
            nonlocal model_logged
            if not model_logged:
                mark_latency("routed_model_received")
                self._log_routed_model(model)
                model_logged = True

        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                completed = False
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        completed = True
                        break

                    yield from self._parse_stream_event(data, log_model)

                if not completed:
                    raise LLMError("OpenRouter stream ended before completion")
        except HTTPError as error:
            print(f"[LLM] http_status={error.code}")
            raise LLMError(f"OpenRouter HTTP {error.code}") from error
        except TimeoutError as error:
            raise LLMError("OpenRouter request timed out") from error
        except URLError as error:
            raise LLMError("OpenRouter connection failed") from error

    def _validate_free_configuration(self) -> None:
        if not self._api_key:
            raise LLMError("OPENROUTER_API_KEY is not configured")
        if not self._is_explicitly_free_model(self._model):
            print("[LLM] model outside exact free allowlist blocked; no request sent")
            raise LLMError("OpenRouter model is not explicitly free and locally approved")

    @staticmethod
    def _is_explicitly_free_model(model: str) -> bool:
        return model in ALLOWED_FREE_MODELS

    def _build_request(
        self, message: str, context: Sequence[ConversationMessage] | None, stream: bool
    ) -> Request:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": self._build_messages(message, context),
            "max_tokens": MAX_RESPONSE_TOKENS,
            "temperature": RESPONSE_TEMPERATURE,
            "provider": {
                "max_price": {"prompt": 0, "completion": 0},
                "data_collection": "deny",
                "allow_fallbacks": True,
            },
            "reasoning": self._reasoning_preferences(),
            "stream": stream,
        }
        return Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
            },
            method="POST",
        )

    def _reasoning_preferences(self) -> dict[str, Any]:
        # Gemma permite desligar thinking; alternativas nao recebem effort nao confirmado.
        preferences: dict[str, Any] = {"exclude": True}
        if self._model == DEFAULT_MODEL:
            preferences["enabled"] = False
        return preferences

    def _log_request(self, stream: bool) -> None:
        print("[LLM] provider=openrouter")
        print(f"[LLM] endpoint={OPENROUTER_URL}")
        print(f"[LLM] model={self._model}")
        print(f"[LLM] streaming={'true' if stream else 'false'}")
        print("[LLM] sending request")

    @staticmethod
    def _build_messages(
        message: str, context: Sequence[ConversationMessage] | None
    ) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": ROCKY_SYSTEM_PROMPT}]
        messages.extend({"role": item.role, "content": item.content} for item in context or ())
        messages.append({"role": "user", "content": message})
        return messages

    @staticmethod
    def _parse_response(response_body: bytes) -> LLMResult:
        try:
            payload: dict[str, Any] = json.loads(response_body)
            choices = payload["choices"]
            content = choices[0]["message"]["content"]
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise LLMError("OpenRouter returned an invalid response") from error

        if not isinstance(content, str) or not content.strip():
            raise LLMError("OpenRouter returned an empty response")

        model = payload.get("model")
        return LLMResult(text=content.strip(), model=model if isinstance(model, str) else None)

    @staticmethod
    def _parse_stream_event(data: str, on_model: Callable[[str], None] | None = None) -> Iterator[str]:
        try:
            payload: dict[str, Any] = json.loads(data)
        except json.JSONDecodeError as error:
            raise LLMError("OpenRouter returned an invalid stream event") from error

        if "error" in payload:
            raise LLMError("OpenRouter returned a stream error")

        model = payload.get("model")
        if isinstance(model, str) and model and on_model is not None:
            on_model(model)

        try:
            content = payload["choices"][0]["delta"].get("content")
        except (IndexError, KeyError, TypeError) as error:
            raise LLMError("OpenRouter returned an invalid stream event") from error

        if content is None:
            return
        if not isinstance(content, str):
            raise LLMError("OpenRouter returned an invalid stream fragment")
        if content:
            yield content

    @staticmethod
    def _log_routed_model(model: str) -> None:
        preview = "".join(character for character in model[:160] if character.isprintable())
        print(f"[LLM] routed_model={preview}")
