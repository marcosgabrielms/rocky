"""Provider Responses API para uso do ChatGPT Plan autenticado localmente."""

from collections.abc import Iterator, Sequence
from typing import Any

from openai import OpenAI

from services.auth import AuthError, ChatGPTSession
from services.llm.base import ConversationMessage, LLMError, LLMResult
from services.llm.personality import ROCKY_SYSTEM_PROMPT
from services.runtime_log import error as log_error, info, technical


DEFAULT_MODEL = "gpt-5.6-terra"
OPENAI_BASE_URL = "https://api.openai.com/v1"
PLAN_ERROR_CODES = frozenset({
    "subscription_sharing_usage_limit_exceeded",
    "subscription_sharing_usage_unavailable",
    "subscription_sharing_user_not_eligible",
    "subscription_sharing_unsupported_capability",
    "subscription_sharing_route_not_supported",
    "subscription_sharing_invalid_user",
    "subscription_sharing_user_unavailable",
    "chatpass_v2_scope_not_authorized",
    "chatpass_v2_invalid_authorization_context",
    "refresh_rotation_pending",
})


class ChatGPTPlanLLMClient:
    """Mantem OAuth e detalhes do SDK fora do restante do fluxo conversacional."""

    def __init__(self, session: ChatGPTSession | None = None, model: str = DEFAULT_MODEL, client_factory=OpenAI) -> None:
        self._session = session or ChatGPTSession()
        self._model = model
        self._client_factory = client_factory

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        text = "".join(self.stream_response(message, context)).strip()
        if not text:
            raise LLMError("ChatGPT Plan returned an empty response")
        return LLMResult(text=text, model=self._model)

    def stream_response(
        self, message: str, context: Sequence[ConversationMessage] | None = None
    ) -> Iterator[str]:
        token = self._access_token()
        client = self._client_factory(api_key=token, base_url=OPENAI_BASE_URL, max_retries=0)
        completed = False
        self._log_request()
        try:
            with client.responses.create(**self._request_arguments(message, context)) as stream:
                for event in stream:
                    event_type = self._event_value(event, "type")
                    if event_type == "response.output_text.delta":
                        delta = self._event_value(event, "delta")
                        if not isinstance(delta, str):
                            raise LLMError("ChatGPT Plan returned an invalid stream delta")
                        if delta:
                            yield delta
                    elif event_type == "response.completed":
                        self._log_usage(self._event_value(event, "response"))
                        completed = True
                    elif event_type == "response.failed":
                        self._raise_event_error(event, "response_failed")
                    elif event_type == "response.incomplete":
                        self._raise_event_error(event, "response_incomplete")
        except LLMError:
            raise
        except Exception as error:
            self._raise_sdk_error(error)

        if not completed:
            raise LLMError("ChatGPT Plan stream ended before response.completed")

    def _access_token(self) -> str:
        try:
            return self._session.access_token_for_inference()
        except AuthError as error:
            log_error("Autenticação ChatGPT necessária.")
            technical(
                f"[LLM] provider=chatgpt_plan auth_error={self._safe_error_code(str(error), 'authentication_required')}"
            )
            raise LLMError("ChatGPT Plan authentication is required") from None

    def _request_arguments(
        self, message: str, context: Sequence[ConversationMessage] | None
    ) -> dict[str, Any]:
        input_items = [{"role": item.role, "content": item.content} for item in context or ()]
        input_items.append({"role": "user", "content": message})
        return {
            "model": self._model,
            "instructions": ROCKY_SYSTEM_PROMPT,
            "input": input_items,
            "store": False,
            "stream": True,
        }

    def _log_request(self) -> None:
        info(f"GPT: {self._model}")
        technical("[LLM] provider=chatgpt_plan")
        technical(f"[LLM] model={self._model}")
        technical("[LLM] streaming=true")
        technical("[LLM] sending request")

    def _raise_event_error(self, event: object, fallback: str) -> None:
        response = self._event_value(event, "response")
        error = self._event_value(response, "error")
        code = self._safe_error_code(self._event_value(error, "code"), fallback)
        log_error("Falha na resposta do ChatGPT.")
        technical(f"[LLM] provider=chatgpt_plan error_code={code}")
        raise LLMError(f"ChatGPT Plan response failed: {code}")

    def _raise_sdk_error(self, error: Exception) -> None:
        code = self._safe_error_code(getattr(error, "code", None), "request_failed")
        log_error("Falha na requisição ao ChatGPT.")
        technical(f"[LLM] provider=chatgpt_plan error_code={code}")
        raise LLMError("ChatGPT Plan request failed") from None

    def _log_usage(self, response: object) -> None:
        usage = self._event_value(response, "usage")
        if usage is None:
            return
        input_tokens = self._usage_value(usage, "input_tokens")
        output_tokens = self._usage_value(usage, "output_tokens")
        total_tokens = self._usage_value(usage, "total_tokens")
        cached_tokens = self._usage_value(self._event_value(usage, "input_tokens_details"), "cached_tokens")
        reasoning_tokens = self._usage_value(self._event_value(usage, "output_tokens_details"), "reasoning_tokens")
        fields = [
            f"model={self._model}",
            f"input={input_tokens}",
            f"output={output_tokens}",
            f"total={total_tokens}",
        ]
        if cached_tokens is not None:
            fields.append(f"cached={cached_tokens}")
        if reasoning_tokens is not None:
            fields.append(f"reasoning={reasoning_tokens}")
        print("[USAGE] " + " ".join(fields))

    @staticmethod
    def _event_value(value: object, name: str) -> Any:
        return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

    @staticmethod
    def _usage_value(value: object, name: str) -> int | None:
        result = ChatGPTPlanLLMClient._event_value(value, name)
        return result if isinstance(result, int) and result >= 0 else None

    @staticmethod
    def _safe_error_code(value: object, fallback: str) -> str:
        return value if isinstance(value, str) and value in PLAN_ERROR_CODES else fallback
