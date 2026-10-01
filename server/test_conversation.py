"""Testes determinísticos para wake word e contexto curto do Rocky."""

import unittest
from collections.abc import Sequence
from unittest.mock import patch

from services.commands import detect_command_intent, normalize_text
from services.conversation import (
    CONTEXT_TTL_SECONDS,
    InteractionKind,
    LLM_INTERRUPTED_MESSAGE,
    LLM_UNAVAILABLE_MESSAGE,
    MAX_CONTEXT_MESSAGES,
    ConversationManager,
)
from services.llm import ConversationMessage, LLMError, LLMResult, OpenRouterLLMClient


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    def ask(self, message: str, context: Sequence[ConversationMessage] | None = None) -> LLMResult:
        self.calls.append((message, tuple(context or ())))
        return LLMResult(text=f"Resposta para: {message}")


class StreamingFakeLLMClient(FakeLLMClient):
    def __init__(self, chunks: tuple[str, ...] = (), error: LLMError | None = None) -> None:
        super().__init__()
        self.chunks = chunks
        self.error = error
        self.stream_calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    def stream_response(self, message: str, context: Sequence[ConversationMessage] | None = None):
        self.stream_calls.append((message, tuple(context or ())))
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


class ConversationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.llm_client = FakeLLMClient()
        self.conversation = ConversationManager(10000, self.llm_client, clock=self.clock)

    def wake(self, device_id: str = "rocky-01", text: str = "Ok, Rocky") -> dict[str, object]:
        return self.conversation.build_response(text, device_id)

    def ask(self, text: str, device_id: str = "rocky-01") -> dict[str, object]:
        self.wake(device_id)
        return self.conversation.build_response(text, device_id)

    def assert_wake(self, text: str) -> None:
        response = self.wake(text=text)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [{"type": "expression", "value": "attention"}])

    def assert_not_wake(self, text: str) -> None:
        response = self.conversation.build_response(text, "idle-test")

        self.assertEqual(response["interaction_state"], "idle")
        self.assertEqual(response["actions"], [])

    def test_accepts_ok_rocky_with_punctuation_and_spacing_variants(self) -> None:
        for text in ("Ok, Rocky", "Ok Rocky", "Ok, Rocky!", "ok rocky"):
            with self.subTest(text=text):
                self.assert_wake(text)

    def test_accepts_rocky_as_temporary_fallback(self) -> None:
        self.assert_wake("Rocky")

    def test_rejects_ok_without_rocky(self) -> None:
        self.assert_not_wake("Ok")

    def test_rejects_common_phrase_with_rocky(self) -> None:
        self.assert_not_wake("Onde está Rocky")

    def test_classify_empty_returns_safe_direct_idle_response(self) -> None:
        decision = self.conversation.classify("", "empty")

        self.assertEqual(decision.kind, InteractionKind.EMPTY)
        self.assertEqual(self.conversation.build_response_from_decision(decision)["interaction_state"], "idle")
        self.assertEqual(self.llm_client.calls, [])

    def test_classify_wake_opens_window_once_and_execution_does_not_reapply_it(self) -> None:
        decision = self.conversation.classify("Ok, Rocky", "wake-once")
        deadline_after_classification = self.conversation._deadlines["wake-once"]

        self.assertEqual(decision.kind, InteractionKind.WAKE)
        response = self.conversation.build_response_from_decision(decision)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(self.conversation._deadlines["wake-once"], deadline_after_classification)
        self.assertEqual(self.llm_client.calls, [])

    def test_classify_deterministic_command_never_calls_llm(self) -> None:
        self.conversation.classify("Ok, Rocky", "hours-decision")
        with patch("services.conversation.get_current_time", return_value="12:00"):
            decision = self.conversation.classify("horas", "hours-decision")
            response = self.conversation.build_response_from_decision(decision)

        self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
        self.assertEqual(response["actions"][0]["type"], "show_text")
        self.assertEqual(self.llm_client.calls, [])

    def test_natural_hours_variants_remain_deterministic_in_both_execution_paths(self) -> None:
        variants = (
            "hora", "horas", "que horas são", "Que horas são?", "que horas sao",
            "que horas é", "que horas e", "qual a hora", "qual é a hora",
            "qual e a hora", "me diga as horas", "me diga a hora",
            "hores", "oras", "ors",
        )
        llm = StreamingFakeLLMClient(("Não deve ser usado.",))
        conversation = ConversationManager(10000, llm, clock=self.clock)
        for text in variants:
            for streaming in (False, True):
                with self.subTest(text=text, streaming=streaming):
                    self.assertEqual(detect_command_intent(normalize_text(text)), "hours")
                    conversation.build_response("Ok, Rocky", "hours-variants")
                    with patch("services.conversation.get_current_time", return_value="12:00"):
                        decision = conversation.classify(text, "hours-variants")
                    self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
                    if streaming:
                        response = conversation.build_stream_response_from_decision(decision).response
                    else:
                        response = conversation.build_response_from_decision(decision)
                    self.assertEqual(response["actions"][0]["line2"], "12:00")
        self.assertEqual(llm.calls, [])
        self.assertEqual(llm.stream_calls, [])

    def test_hours_matching_does_not_accept_substrings_or_arbitrary_phrases(self) -> None:
        for text in ("horário", "melhoras", "horas extras", "quantas horas tem um dia", "me diga a história"):
            with self.subTest(text=text):
                self.assertIsNone(detect_command_intent(normalize_text(text)))

    def test_llm_decision_is_executed_once_without_reclassification(self) -> None:
        self.conversation.classify("Ok, Rocky", "one-llm")
        with patch.object(self.conversation, "classify", wraps=self.conversation.classify) as classify:
            decision = self.conversation.classify("Pergunta aberta", "one-llm")
            response = self.conversation.build_response_from_decision(decision)

        self.assertEqual(decision.kind, InteractionKind.LLM)
        self.assertEqual(classify.call_count, 1)
        self.assertEqual(len(self.llm_client.calls), 1)
        self.assertEqual(response["interaction_state"], "idle")

    def test_stream_llm_decision_is_executed_once_without_reclassification(self) -> None:
        llm_client = StreamingFakeLLMClient(("Resposta.",))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)
        conversation.classify("Ok, Rocky", "one-stream")
        with patch.object(conversation, "classify", wraps=conversation.classify) as classify:
            decision = conversation.classify("Pergunta aberta", "one-stream")
            result = conversation.build_stream_response_from_decision(decision)

        self.assertEqual(decision.kind, InteractionKind.LLM)
        self.assertEqual(classify.call_count, 1)
        self.assertEqual(len(llm_client.stream_calls), 1)
        self.assertEqual(result.response["text"], "Resposta.")

    def test_second_question_receives_context_from_first_question(self) -> None:
        self.ask("Quem criou o Linux?")
        self.clock.now += 1
        self.ask("E de onde ele é?")

        self.assertEqual(
            self.llm_client.calls[-1][1],
            (
                ConversationMessage(role="user", content="Quem criou o Linux?"),
                ConversationMessage(role="assistant", content="Resposta para: Quem criou o Linux?"),
            ),
        )

    def test_context_is_separated_by_device_id(self) -> None:
        self.ask("Pergunta do primeiro Rocky", "rocky-01")
        self.clock.now += 1
        self.ask("Pergunta do segundo Rocky", "rocky-02")

        self.assertEqual(self.llm_client.calls[-1][1], ())

    def test_context_keeps_at_most_six_messages(self) -> None:
        for index in range(4):
            self.ask(f"Pergunta {index}")
            self.clock.now += 1

        history = self.conversation._histories["rocky-01"].messages
        self.assertEqual(len(history), MAX_CONTEXT_MESSAGES)
        self.assertEqual(history[0].content, "Pergunta 1")
        self.assertEqual(history[-1].content, "Resposta para: Pergunta 3")

    def test_context_expires_after_five_minutes_without_valid_interaction(self) -> None:
        self.ask("Primeira pergunta")
        self.clock.now += CONTEXT_TTL_SECONDS + 1
        self.ask("Pergunta após expiração")

        self.assertEqual(self.llm_client.calls[-1][1], ())

    def test_wake_does_not_enter_history(self) -> None:
        self.wake()

        self.assertNotIn("rocky-01", self.conversation._histories)

    def test_hours_does_not_enter_history(self) -> None:
        self.wake()
        with patch("services.conversation.get_current_time", return_value="12:00"):
            self.conversation.build_response("horas", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_question_in_idle_does_not_enter_history(self) -> None:
        self.conversation.build_response("Quem criou o Linux?", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_empty_text_does_not_enter_history(self) -> None:
        self.wake()
        self.conversation.build_response("", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_openrouter_preserves_context_roles(self) -> None:
        messages = OpenRouterLLMClient._build_messages(
            "Nova pergunta",
            (
                ConversationMessage(role="user", content="Pergunta anterior"),
                ConversationMessage(role="assistant", content="Resposta anterior"),
            ),
        )

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])

    def test_streaming_question_uses_one_generation_and_returns_sentences(self) -> None:
        llm_client = StreamingFakeLLMClient(("Olá. Como", " posso ajudar?"))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)

        conversation.build_stream_response("Ok, Rocky", "streaming-rocky")
        result = conversation.build_stream_response("Me conte algo", "streaming-rocky")

        self.assertEqual(len(llm_client.stream_calls), 1)
        self.assertEqual(llm_client.calls, [])
        self.assertEqual(result.sentences, ("Olá.", "Como posso ajudar?"))
        self.assertEqual(result.response["text"], "Olá. Como posso ajudar?")
        self.assertEqual(result.response["interaction_state"], "idle")
        self.assertEqual(result.response["actions"], [])

    def test_free_model_errors_before_first_token_return_controlled_response(self) -> None:
        for status_code in (402, 429):
            with self.subTest(status_code=status_code):
                llm_client = StreamingFakeLLMClient(error=LLMError(f"OpenRouter HTTP {status_code}"))
                conversation = ConversationManager(10000, llm_client, clock=self.clock)

                conversation.build_stream_response("Ok, Rocky", f"streaming-error-{status_code}")
                result = conversation.build_stream_response("Pergunta", f"streaming-error-{status_code}")

                self.assertEqual(len(llm_client.stream_calls), 1)
                self.assertEqual(result.sentences, ())
                self.assertEqual(result.response["text"], LLM_UNAVAILABLE_MESSAGE)

    def test_stream_error_after_partial_text_never_retries(self) -> None:
        llm_client = StreamingFakeLLMClient(("Resposta parcial"), LLMError("OpenRouter HTTP 429"))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)

        conversation.build_stream_response("Ok, Rocky", "streaming-partial")
        result = conversation.build_stream_response("Pergunta", "streaming-partial")

        self.assertEqual(len(llm_client.stream_calls), 1)
        self.assertEqual(result.sentences, ("Resposta parcial",))
        self.assertEqual(result.response["text"], f"Resposta parcial {LLM_INTERRUPTED_MESSAGE}")

    def test_streaming_hours_remains_deterministic_without_llm(self) -> None:
        llm_client = StreamingFakeLLMClient(("Não deve ser usado.",))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)

        conversation.build_stream_response("Ok, Rocky", "streaming-hours")
        with patch("services.conversation.get_current_time", return_value="12:00"):
            result = conversation.build_stream_response("horas", "streaming-hours")

        self.assertEqual(llm_client.stream_calls, [])
        self.assertEqual(result.response["actions"][0]["type"], "show_text")


if __name__ == "__main__":
    unittest.main()
