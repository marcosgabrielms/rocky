"""Provider Gemini Developer API (plano gratuito do AI Studio)."""

import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from services.llm.base import ConversationMessage, LLMError, LLMResult
from services.llm.personality import ROCKY_SYSTEM_PROMPT
from services.runtime_log import error as log_error, info, technical


API_KEY_ENV = "GEMINI_API_KEY"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_MS = 20_000
# Códigos HTTP que o log pode exibir sem expor corpo de resposta.
SAFE_STATUS_CODES = frozenset({400, 401, 403, 404, 408, 429, 500, 502, 503, 504})


def _default_client_factory(**kwargs: Any) -> Any:
    from google import genai

    return genai.Client(**kwargs)


class GeminiLLMClient:
    """Mantém detalhes do SDK google-genai fora do fluxo conversacional."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client_factory: Callable[..., Any] = _default_client_factory,
        timeout_ms: int = REQUEST_TIMEOUT_MS,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_ms = timeout_ms
        self._client_factory = client_factory
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        text = "".join(self.stream_response(message, context)).strip()
        if not text:
            raise LLMError("Gemini returned an empty response")
        return LLMResult(text=text, model=self._model)

    def stream_response(
        self, message: str, context: Sequence[ConversationMessage] | None = None
    ) -> Iterator[str]:
        client = self._get_client()
        system_instruction, contents = self._build_request(message, context)
        self._log_request()
        last_chunk: Any = None
        try:
            stream = client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config={"system_instruction": system_instruction},
            )
            for chunk in stream:
                last_chunk = chunk
                text = self._chunk_text(chunk)
                if text:
                    yield text
        except LLMError:
            raise
        except Exception as error:
            self._raise_sdk_error(error)
        self._log_usage(last_chunk)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = self._api_key or os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            log_error("Chave do Gemini ausente.")
            technical(f"[LLM] provider=gemini error_code=api_key_missing model={self._model}")
            raise LLMError("Gemini API key is missing")
        self._client = self._client_factory(api_key=api_key, http_options={"timeout": self._timeout_ms})
        return self._client

    @staticmethod
    def _build_request(
        message: str, context: Sequence[ConversationMessage] | None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Gemini só aceita user/model no histórico; system vai para a instrução."""
        system_parts = [ROCKY_SYSTEM_PROMPT]
        contents: list[dict[str, Any]] = []
        for item in context or ():
            if item.role == "system":
                system_parts.append(item.content)
                continue
            role = "model" if item.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": item.content}]})
        contents.append({"role": "user", "parts": [{"text": message}]})
        return "\n\n".join(system_parts), contents

    @staticmethod
    def _chunk_text(chunk: Any) -> str:
        candidates = getattr(chunk, "candidates", None) or ()
        pieces: list[str] = []
        for candidate in candidates[:1]:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or ():
                text = getattr(part, "text", None)
                if isinstance(text, str) and text and not getattr(part, "thought", False):
                    pieces.append(text)
        return "".join(pieces)

    def _log_request(self) -> None:
        info(f"Gemini: {self._model}")
        technical("[LLM] provider=gemini")
        technical(f"[LLM] model={self._model}")
        technical("[LLM] streaming=true")
        technical("[LLM] sending request")

    def _raise_sdk_error(self, error: Exception) -> None:
        status = getattr(error, "code", None)
        if status == 429:
            code = "quota_exhausted"
            log_error("Cota do Gemini esgotada.")
        else:
            code = f"http_{status}" if status in SAFE_STATUS_CODES else "request_failed"
            log_error("Falha na requisição ao Gemini.")
        technical(f"[LLM] provider=gemini model={self._model} error_code={code}")
        raise LLMError(f"Gemini request failed: {code}") from None

    def _log_usage(self, chunk: Any) -> None:
        usage = getattr(chunk, "usage_metadata", None)
        if usage is None:
            return
        fields = [f"model={self._model}"]
        for label, name in (
            ("input", "prompt_token_count"),
            ("output", "candidates_token_count"),
            ("total", "total_token_count"),
            ("cached", "cached_content_token_count"),
            ("reasoning", "thoughts_token_count"),
        ):
            value = getattr(usage, name, None)
            if isinstance(value, int) and value >= 0:
                fields.append(f"{label}={value}")
        print("[USAGE] " + " ".join(fields))
