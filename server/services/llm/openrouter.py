"""Cliente HTTP mínimo para a API Chat Completions do OpenRouter."""

import json
import os
import socket
import time
from http.client import HTTPException
from threading import Timer
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.llm.base import ConversationMessage, LLMError, LLMResult
from services.llm.personality import ROCKY_SYSTEM_PROMPT
from services.latency import mark_latency


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "inclusionai/ling-3.1-flash"
GEMMA_MODEL = "google/gemma-4-31b-it:free"
ALLOWED_FREE_MODELS = frozenset({DEFAULT_MODEL, GEMMA_MODEL})
REQUEST_TIMEOUT_SECONDS = 20
STREAM_DEADLINE_SECONDS = 120
MAX_RESPONSE_TOKENS = 120
RESPONSE_TEMPERATURE = 0.2
MAX_ERROR_BODY_BYTES = 8192
SAFE_ERROR_CODES = frozenset({
    "rate_limit_exceeded", "rate_limited", "provider_overloaded", "provider_unavailable",
    "context_length_exceeded", "max_tokens_exceeded", "token_limit_exceeded", "string_too_long",
    "authentication", "permission_denied", "payment_required", "insufficient_quota",
    "invalid_request", "invalid_prompt", "not_found", "precondition_failed",
    "payload_too_large", "unprocessable", "content_policy_violation", "refusal",
    "internal_server_error", "server_error", "provider_timeout",
})
SAFE_PROVIDER_NAMES = frozenset({
    "Google", "Google AI Studio", "Google Vertex", "NVIDIA", "Chutes",
    "DeepInfra", "Together", "Fireworks", "Novita", "SiliconFlow",
})
SAFE_ROUTED_MODELS = ALLOWED_FREE_MODELS | frozenset(
    model.removesuffix(":free") for model in ALLOWED_FREE_MODELS
)


class _StreamDeadline:
    """Limita tambem uma leitura SSE bloqueada recebendo bytes sem newline."""

    def __init__(self, response: Any, expires_at: float) -> None:
        self._response = response
        self._expires_at = expires_at
        self._timer = Timer(max(0, expires_at - time.monotonic()), self._abort)
        self._timer.daemon = True

    def __enter__(self) -> "_StreamDeadline":
        self.check()
        self._timer.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._timer.cancel()

    def check(self) -> None:
        if time.monotonic() >= self._expires_at:
            raise LLMError("OpenRouter stream total deadline exceeded")

    def _abort(self) -> None:
        # HTTPResponse.close pode esperar pelo lock da leitura; shutdown a desbloqueia.
        raw = getattr(getattr(self._response, "fp", None), "raw", None)
        connection = getattr(raw, "_sock", None)
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


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
            self._log_http_error(error)
            raise LLMError(f"OpenRouter HTTP {error.code}") from error
        except TimeoutError as error:
            raise LLMError("OpenRouter request timed out") from error
        except URLError as error:
            raise LLMError("OpenRouter connection failed") from error
        except (OSError, HTTPException) as error:
            raise LLMError("OpenRouter response interrupted") from error

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
        expires_at = time.monotonic() + STREAM_DEADLINE_SECONDS
        model_logged = False

        def log_model(model: str) -> None:
            nonlocal model_logged
            if not model_logged:
                mark_latency("routed_model_received")
                self._log_routed_model(model)
                model_logged = True

        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response, _StreamDeadline(response, expires_at) as deadline:
                completed = False
                for raw_line in response:
                    deadline.check()
                    line = raw_line.decode("utf-8").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue

                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        completed = True
                        break

                    yield from self._parse_stream_event(data, log_model)
                    deadline.check()

                deadline.check()
                if not completed:
                    raise LLMError("OpenRouter stream ended before completion")
        except HTTPError as error:
            self._log_http_error(error)
            raise LLMError(f"OpenRouter HTTP {error.code}") from error
        except TimeoutError as error:
            raise LLMError("OpenRouter request timed out") from error
        except URLError as error:
            raise LLMError("OpenRouter connection failed") from error
        except (OSError, HTTPException) as error:
            raise LLMError("OpenRouter response interrupted") from error

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
        # Ling usa apenas exclude; a preferencia anterior do override Gemma permanece.
        preferences: dict[str, Any] = {"exclude": True}
        if self._model == GEMMA_MODEL:
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
            if isinstance(payload, dict) and "error" in payload:
                OpenRouterLLMClient._log_error_details(payload)
                raise LLMError("OpenRouter returned an error response")
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
            OpenRouterLLMClient._log_error_details(payload)
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
        # O campo remoto tambem pode conter texto arbitrario; nao ecoar valores desconhecidos.
        print(f"[LLM] routed_model={model if model in SAFE_ROUTED_MODELS else '[unrecognized]'}")

    @classmethod
    def _log_http_error(cls, error: HTTPError) -> None:
        print(f"[LLM] http_status={error.code}")
        try:
            with error:
                body = error.read(MAX_ERROR_BODY_BYTES + 1)
            if len(body) > MAX_ERROR_BODY_BYTES:
                print("[LLM] error_body=too_large; details omitted")
                return
            cls._log_error_details(json.loads(body))
        except (OSError, HTTPException, ValueError, TypeError, RecursionError):
            print("[LLM] error_body=unavailable_or_invalid; details omitted")

    @classmethod
    def _log_error_details(cls, payload: Any) -> None:
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return
        code = cls._safe_error_code(error.get("code"))
        if code is not None:
            print(f"[LLM] error_code={code}")
        message = error.get("message")
        if isinstance(message, str) and message:
            print(f"[LLM] error_message={json.dumps(cls._safe_error_message(message))}")
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            cls._log_safe_metadata(metadata)

    @staticmethod
    def _safe_error_code(value: Any) -> str | int | None:
        if type(value) is int and 100 <= value <= 599:
            return value
        if isinstance(value, str) and value in SAFE_ERROR_CODES:
            return value
        return None

    @staticmethod
    def _safe_error_message(message: str) -> str:
        # Projecao fechada: nunca imprimir trechos livres, mesmo truncados, do corpo remoto.
        normalized = message[:MAX_ERROR_BODY_BYTES].lower()
        if "rate limit" in normalized or "rate_limit" in normalized or "rate-limit" in normalized:
            scope = next((name for name in ("account", "model", "provider", "upstream")
                          if name in normalized), "unspecified")
            return f"Rate limit reported; scope={scope}."
        categories = (
            (("overloaded", "capacity"), "Capacity limit reported."),
            (("data policy", "data collection", "privacy"), "Provider privacy policy incompatibility reported."),
            (("no endpoints", "unavailable"), "No eligible endpoint or service unavailable reported."),
            (("quota", "credits", "payment"), "Quota or credit limit reported."),
            (("guardrail", "policy violation"), "Policy restriction reported."),
            (("authentication", "unauthorized", "api key"), "Authentication failure reported."),
            (("context length", "token limit"), "Token or context limit reported."),
        )
        for patterns, safe_message in categories:
            if any(pattern in normalized for pattern in patterns):
                return safe_message
        return "Unrecognized error message; details omitted."

    @classmethod
    def _log_safe_metadata(cls, metadata: dict[str, Any]) -> None:
        for field in ("error_type", "provider_code"):
            value = cls._safe_error_code(metadata.get(field))
            if value is not None:
                print(f"[LLM] metadata.{field}={value}")
        provider = metadata.get("provider_name")
        if isinstance(provider, str) and provider in SAFE_PROVIDER_NAMES:
            print(f"[LLM] metadata.provider_name={provider}")
        model = metadata.get("model")
        if isinstance(model, str) and model in SAFE_ROUTED_MODELS:
            print(f"[LLM] metadata.model={model}")
