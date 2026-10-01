"""Cliente HTTP mínimo para a API Chat Completions do OpenRouter."""

import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.llm.base import ConversationMessage, LLMError, LLMResult
from services.llm.personality import ROCKY_SYSTEM_PROMPT


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"
ALLOWED_FREE_MODELS = frozenset({"openrouter/free", "openai/gpt-oss-20b:free"})
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESPONSE_TOKENS = 160


class OpenRouterLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

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
            print(f"[LLM] response_model={result.model}")
        return result

    def stream_response(
        self, message: str, context: Sequence[ConversationMessage] | None = None
    ) -> Iterator[str]:
        """Produz os fragmentos SSE de uma única geração OpenRouter."""
        self._validate_free_configuration()
        request = self._build_request(message, context, stream=True)
        self._log_request(stream=True)

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

                    yield from self._parse_stream_event(data)

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
            print("[LLM] paid/non-approved model blocked")
            raise LLMError("OpenRouter model is not explicitly free")

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
    def _parse_stream_event(data: str) -> Iterator[str]:
        try:
            payload: dict[str, Any] = json.loads(data)
        except json.JSONDecodeError as error:
            raise LLMError("OpenRouter returned an invalid stream event") from error

        if "error" in payload:
            raise LLMError("OpenRouter returned a stream error")

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
