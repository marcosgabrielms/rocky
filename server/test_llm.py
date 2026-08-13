"""Teste manual do fluxo OpenRouter integrado à conversa do Rocky."""

import sys

from services.conversation import ConversationManager
from services.llm import LLMResult, OpenRouterLLMClient


QUESTION = "Responda em português: por que o céu é azul?"
DEVICE_ID = "manual-openrouter-test"
PERSONALITY_QUESTIONS = (
    "Quanto é 2 + 2?",
    "Por que o céu é azul?",
    "Você é inteligente?",
    "Explique com mais detalhes por que o céu é azul.",
    "Quanto é 10% de 200?",
)


class CountingLLMClient:
    def __init__(self, client: OpenRouterLLMClient) -> None:
        self._client = client
        self.calls = 0

    def ask(self, message: str, context: list[str] | None = None) -> LLMResult:
        self.calls += 1
        return self._client.ask(message, context)


def main() -> None:
    llm_client = CountingLLMClient(OpenRouterLLMClient())
    conversation = ConversationManager(10000, llm_client)

    wake = conversation.build_response("Rocky", DEVICE_ID)
    assert wake["interaction_state"] == "attention"
    assert wake["command_window_ms"] == 10000
    assert wake["actions"] == [{"type": "expression", "value": "attention"}]
    print("[TEST] wake=attention")

    response = conversation.build_response(QUESTION, DEVICE_ID)
    assert response["interaction_state"] == "idle"
    assert response["actions"] == []
    assert response["text"]
    assert llm_client.calls == 1
    print("[TEST] llm_called=yes")
    print(f"[TEST] backend_text={response['text']}")

    idle_response = conversation.build_response(QUESTION, "idle-openrouter-test")
    assert idle_response["interaction_state"] == "idle"
    assert idle_response["actions"] == []
    assert llm_client.calls == 1
    print("[TEST] idle_question_llm_called=no")

    hours_wake = conversation.build_response("Rocky", "hours-openrouter-test")
    assert hours_wake["interaction_state"] == "attention"
    hours = conversation.build_response("horas", "hours-openrouter-test")
    assert hours["interaction_state"] == "idle"
    assert hours["actions"][0]["type"] == "show_text"
    assert hours["actions"][0]["line1"] == "Agora sao"
    assert llm_client.calls == 1
    print("[TEST] hours_llm_called=no")

    if "--personality" in sys.argv:
        print("[TEST] personality responses")
        for question in PERSONALITY_QUESTIONS:
            result = llm_client.ask(question)
            print(f"Pergunta: {question}")
            print(f"Resposta: {result.text}")
            if result.model:
                print(f"Modelo: {result.model}")

    print("[TEST] integration=OK")


if __name__ == "__main__":
    main()
