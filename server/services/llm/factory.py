"""Escolhe o cérebro do Koda a partir da configuração local (.env)."""

from collections.abc import Mapping

from services.llm.base import StreamingLLMClient
from services.llm.fallback import FallbackLLMClient
from services.llm.gemini import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL, GeminiLLMClient


PROVIDER_ENV = "KODA_LLM_PROVIDER"
GEMINI_MODEL_ENV = "KODA_GEMINI_MODEL"
GEMINI_FALLBACK_MODELS_ENV = "KODA_GEMINI_FALLBACK_MODELS"
DEFAULT_PROVIDER = "gemini"
# Cada modelo tem cota diária própria no plano gratuito: 3.1 Flash Lite cobre
# picos/erros 503 do principal; Gemma (14.4K/dia) é o último recurso.
DEFAULT_GEMINI_FALLBACK_MODELS = ("gemini-3.1-flash-lite", "gemma-4-31b-it")
# Gemma sempre raciocina antes de responder (medido: 8-47 s no plano gratuito).
SLOW_MODEL_PREFIX = "gemma-"
SLOW_MODEL_TIMEOUT_MS = 90_000
SUPPORTED_PROVIDERS = frozenset({"gemini", "chatgpt_plan"})


def selected_provider(env: Mapping[str, str]) -> str:
    provider = env.get(PROVIDER_ENV, "").strip().lower() or DEFAULT_PROVIDER
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported {PROVIDER_ENV}: {provider}")
    return provider


def gemini_models(env: Mapping[str, str]) -> tuple[str, ...]:
    primary = env.get(GEMINI_MODEL_ENV, "").strip() or GEMINI_DEFAULT_MODEL
    # Variável definida e vazia desliga a reserva; ausente usa o padrão.
    raw = env.get(GEMINI_FALLBACK_MODELS_ENV)
    fallbacks = DEFAULT_GEMINI_FALLBACK_MODELS if raw is None else tuple(raw.split(","))
    models = [primary]
    for model in (item.strip() for item in fallbacks):
        if model and model not in models:
            models.append(model)
    return tuple(models)


def _gemini_client(model: str) -> GeminiLLMClient:
    if model.startswith(SLOW_MODEL_PREFIX):
        return GeminiLLMClient(model=model, timeout_ms=SLOW_MODEL_TIMEOUT_MS)
    return GeminiLLMClient(model=model)


def build_llm_client(env: Mapping[str, str]) -> StreamingLLMClient:
    provider = selected_provider(env)
    if provider == "chatgpt_plan":
        from services.llm.chatgpt_plan import ChatGPTPlanLLMClient

        return ChatGPTPlanLLMClient()

    clients = [_gemini_client(model) for model in gemini_models(env)]
    return clients[0] if len(clients) == 1 else FallbackLLMClient(clients)
