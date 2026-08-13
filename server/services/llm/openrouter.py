"""Cliente HTTP mínimo para a API Chat Completions do OpenRouter."""

import json
import os
import time
from collections.abc import Sequence
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.llm.base import LLMError, LLMResult
from services.llm.personality import ROCKY_SYSTEM_PROMPT


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openrouter/free"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RESPONSE_TOKENS = 160


class OpenRouterLLMClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self._model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

    def ask(self, message: str, context: Sequence[str] | None = None) -> LLMResult:
        if not self._api_key:
            raise LLMError("OPENROUTER_API_KEY is not configured")

        payload = {
            "model": self._model,
            "messages": self._build_messages(message, context),
            "max_tokens": MAX_RESPONSE_TOKENS,
        }
        request = Request(
            OPENROUTER_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        print("[LLM] provider=openrouter")
        print(f"[LLM] model={self._model}")
        print("[LLM] sending request")
        started_at = time.monotonic()
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response_body = response.read()
        except HTTPError as error:
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

    @staticmethod
    def _build_messages(message: str, context: Sequence[str] | None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": ROCKY_SYSTEM_PROMPT}]
        messages.extend({"role": "user", "content": item} for item in context or ())
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
