"""Encadeia provedores: o próximo só assume se o anterior falhar antes de falar."""

from collections.abc import Iterator, Sequence

from services.llm.base import ConversationMessage, LLMError, LLMResult, StreamingLLMClient
from services.runtime_log import technical


class FallbackLLMClient:
    """Tenta cada cliente em ordem; nunca mistura duas respostas na mesma fala."""

    def __init__(self, clients: Sequence[StreamingLLMClient]) -> None:
        if not clients:
            raise ValueError("FallbackLLMClient requires at least one client")
        self._clients = tuple(clients)

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        last_error: LLMError | None = None
        for index, client in enumerate(self._clients):
            try:
                return client.ask(message, context)
            except LLMError as error:
                last_error = error
                self._log_switch(index)
        raise last_error or LLMError("No LLM client available")

    def stream_response(
        self, message: str, context: Sequence[ConversationMessage] | None = None
    ) -> Iterator[str]:
        last_error: LLMError | None = None
        for index, client in enumerate(self._clients):
            yielded = False
            try:
                for chunk in client.stream_response(message, context):
                    if chunk:
                        yielded = True
                        yield chunk
                return
            except LLMError as error:
                if yielded:
                    raise
                last_error = error
                self._log_switch(index)
        raise last_error or LLMError("No LLM client available")

    def _log_switch(self, index: int) -> None:
        if index + 1 < len(self._clients):
            technical(f"[LLM] fallback from_index={index} to_index={index + 1}")
