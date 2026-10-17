"""Testes determinísticos para wake word e contexto curto do Rocky."""

import io
import os
import unittest
from collections.abc import Sequence
from contextlib import redirect_stdout
from unittest.mock import patch

from services.commands import detect_command_intent, extract_volume_percent, normalize_text
from services.conversation import (
    CONTEXT_TTL_SECONDS,
    InteractionKind,
    InteractionMode,
    LLM_INTERRUPTED_MESSAGE,
    LLM_UNAVAILABLE_MESSAGE,
    MAX_CONTEXT_MESSAGES,
    ConversationManager,
    canonical_wake_word,
)
from services.command_session import CommandSession
from services.llm import ConversationMessage, LLMError, LLMResult, OpenRouterLLMClient
from services.latency import LatencyTrace, latency_scope


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

    def wake(self, device_id: str = "rocky-01", text: str = "Koda") -> dict[str, object]:
        return self.conversation.build_response(text, device_id, interaction_mode=InteractionMode.WAKE)

    def ask(self, text: str, device_id: str = "rocky-01") -> dict[str, object]:
        self.wake(device_id)
        return self.conversation.build_response(text, device_id, interaction_mode=InteractionMode.COMMAND)

    def assert_wake(self, text: str) -> None:
        response = self.wake(text=text)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [{"type": "expression", "value": "attention"}])

    def assert_not_wake(self, text: str) -> None:
        response = self.conversation.build_response(text, "idle-test")

        self.assertEqual(response["interaction_state"], "idle")
        self.assertEqual(response["actions"], [])

    def test_accepts_koda_without_ok_and_with_punctuation(self) -> None:
        for text in ("Koda", "Koda.", "Koda?", "koda", "KODA", "Kôda", "KÔDA", "  Koda!  "):
            with self.subTest(text=text):
                self.assert_wake(text)

    def test_koda_and_koda_with_circumflex_share_the_same_canonical_wake(self) -> None:
        self.assertEqual(canonical_wake_word("Koda"), "koda")
        self.assertEqual(canonical_wake_word("Kôda"), "koda")
        self.assertEqual(canonical_wake_word("KÔDA"), "koda")

    def test_rejects_legacy_wake_and_similar_words(self) -> None:
        for text in ("Rocky", "Ok Rocky", "Ok Koda", "Ei Koda", "alguma coisa", "Coda", "Calma",
                     "Kodal", "Koda123", "Koda_nome", "Eu falei Koda ontem.", "O nome dele é Koda."):
            with self.subTest(text=text):
                self.assert_not_wake(text)

    def test_same_utterance_hours_extracts_command_without_llm_or_attention(self) -> None:
        for text in ("Koda, que horas são?", "KODA! Que horas são?", "Koda horas", "  \"Koda\", horas."):
            with self.subTest(text=text), patch("services.conversation.get_current_time", return_value="12:00"):
                decision = self.conversation.classify(text, "combined", interaction_mode=InteractionMode.WAKE)
                response = self.conversation.build_response_from_decision(decision)
                self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
                self.assertEqual(response["interaction_state"], "idle")
                self.assertEqual(response["actions"][0]["line2"], "12:00")
                self.assertNotIn("combined", self.conversation.context_store._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_sleep_command_is_deterministic_and_never_calls_the_llm(self) -> None:
        for text in (
            "Koda, vai dormir", "Koda, pode dormir", "Koda, dorme", "Koda, vá dormir",
            "Koda, descansa", "pode dormir, Koda",
        ):
            with self.subTest(text=text):
                decision = self.conversation.classify(text, "sleep-test", interaction_mode=InteractionMode.WAKE)
                response = self.conversation.build_response_from_decision(decision)

                self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
                self.assertEqual(response["text"], "Vou dormir.")
                self.assertEqual(response["actions"], [{"type": "set_device_state", "state": "sleeping"}])
        self.assertEqual(self.llm_client.calls, [])

    def test_wake_device_command_is_deterministic_and_never_calls_the_llm(self) -> None:
        for text in ("Koda, acorda", "Koda pode acordar", "acorda, Koda"):
            with self.subTest(text=text):
                decision = self.conversation.classify(text, "wake-device", interaction_mode=InteractionMode.WAKE)
                response = self.conversation.build_response_from_decision(decision)

                self.assertEqual(response["text"], "Ja estou acordado.")
                self.assertEqual(response["interaction_state"], "attention")
                self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm_client.calls, [])

    def test_any_valid_question_after_koda_reaches_llm_once_without_wake_in_history(self) -> None:
        questions = (
            "Quem criou o Linux?", "Quem escreveu Dom Casmurro?", "Por que o céu é azul?",
            "O que é fotossíntese?", "Quem foi Alan Turing?", "Qual é o maior planeta do Sistema Solar?",
            "O que é recursão?", "Como funciona um eclipse?", "Por que o mar é salgado?",
        )
        for index, question in enumerate(questions):
            with self.subTest(question=question):
                device = f"question-{index}"
                decision = self.conversation.classify(f"Koda, {question}", device, interaction_mode=InteractionMode.WAKE)
                self.assertEqual(decision.kind, InteractionKind.LLM)
                self.assertEqual(decision.text, question)
                response = self.conversation.build_response_from_decision(decision)
                self.assertEqual(len(self.llm_client.calls), index + 1)
                self.assertEqual(self.llm_client.calls[-1][0], question)
                self.assertEqual(response["interaction_state"], "idle")
                self.assertEqual(self.conversation.context_store._histories[device].messages[0].content, question)

    def test_same_utterance_streams_once_and_removes_only_initial_wake(self) -> None:
        llm = StreamingFakeLLMClient(("Resposta curta.",))
        manager = ConversationManager(0, llm, clock=self.clock)
        result = manager.build_stream_response("Koda quem criou o Linux?", "combined-stream")
        self.assertEqual(llm.stream_calls, [("quem criou o Linux?", ())])
        self.assertEqual(llm.calls, [])
        self.assertEqual(result.response["text"], "Resposta curta.")

    def test_passive_idle_speech_does_not_invoke_llm_or_create_session(self) -> None:
        for text in ("bom dia pessoal", "Ok Koda", "Eu falei Koda ontem", "Oi", "Kodal"):
            response = self.conversation.build_response(text, "passive", interaction_mode=InteractionMode.WAKE)
            self.assertEqual(response["interaction_state"], "idle")
            self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm_client.calls, [])

    def test_inline_wake_does_not_bypass_mode_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.conversation.classify("Koda, pergunta", "invalid", interaction_mode="invalid")

    def test_rejects_ok_without_rocky(self) -> None:
        self.assert_not_wake("Ok")

    def test_rejects_common_phrase_with_rocky(self) -> None:
        self.assert_not_wake("Onde está Koda")

    def test_classify_empty_returns_safe_direct_idle_response(self) -> None:
        decision = self.conversation.classify("", "empty")

        self.assertEqual(decision.kind, InteractionKind.EMPTY)
        self.assertEqual(self.conversation.build_response_from_decision(decision)["interaction_state"], "idle")
        self.assertEqual(self.llm_client.calls, [])

    def test_classify_wake_does_not_own_command_session(self) -> None:
        decision = self.conversation.classify("Koda", "wake-once", interaction_mode=InteractionMode.WAKE)

        self.assertEqual(decision.kind, InteractionKind.WAKE)
        response = self.conversation.build_response_from_decision(decision)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertFalse(hasattr(self.conversation, "has_active_command_window"))
        self.assertFalse(hasattr(self.conversation, "_waiting_commands"))
        self.assertEqual(self.llm_client.calls, [])

    def test_classify_deterministic_command_never_calls_llm(self) -> None:
        self.conversation.classify("Koda", "hours-decision", interaction_mode=InteractionMode.WAKE)
        with patch("services.conversation.get_current_time", return_value="12:00"):
            decision = self.conversation.classify("horas", "hours-decision", interaction_mode=InteractionMode.COMMAND)
            response = self.conversation.build_response_from_decision(decision)

        self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
        self.assertEqual(response["actions"][0]["type"], "show_text")
        self.assertEqual(self.llm_client.calls, [])

    def test_non_speech_after_wake_preserves_attention_without_calling_llm(self) -> None:
        self.wake()

        response = self.conversation.build_response("", "rocky-01", interaction_mode=InteractionMode.COMMAND)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm_client.calls, [])

    def test_filler_after_wake_preserves_attention_without_calling_llm(self) -> None:
        self.wake()

        response = self.conversation.build_response("aham", "rocky-01", interaction_mode=InteractionMode.COMMAND)

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm_client.calls, [])

    def test_low_confidence_post_wake_text_preserves_attention_without_calling_llm(self) -> None:
        self.wake()

        response = self.conversation.build_response(
            "e", "rocky-01", interaction_mode=InteractionMode.COMMAND, low_confidence=True,
        )

        self.assertEqual(response["interaction_state"], "attention")
        self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm_client.calls, [])

    def test_non_lexical_vocalization_after_wake_preserves_attention_without_calling_llm(self) -> None:
        for text in ("kkkkkkkkkkkk", "aaaaaaa", "mmmmmmmm", "hhhhhhhh"):
            with self.subTest(text=text):
                self.wake()
                response = self.conversation.build_response(text, "rocky-01", interaction_mode=InteractionMode.COMMAND)

                self.assertEqual(response["interaction_state"], "attention")
                self.assertEqual(response["actions"], [])
                self.assertEqual(self.llm_client.calls, [])

    def test_valid_single_word_command_after_wake_remains_available(self) -> None:
        self.wake()

        response = self.conversation.build_response("azul", "rocky-01", interaction_mode=InteractionMode.COMMAND)

        self.assertEqual(response["interaction_state"], "idle")
        self.assertEqual(self.llm_client.calls, [("azul", ())])

    def test_confident_question_after_wake_remains_eligible_for_llm(self) -> None:
        self.wake()

        response = self.conversation.build_response(
            "Qual e a capital do Brasil?", "rocky-01", interaction_mode=InteractionMode.COMMAND,
            low_confidence=False,
        )

        self.assertEqual(response["interaction_state"], "idle")
        self.assertEqual(self.llm_client.calls, [("Qual e a capital do Brasil?", ())])

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
                    conversation.build_response("Koda", "hours-variants", interaction_mode=InteractionMode.WAKE)
                    with patch("services.conversation.get_current_time", return_value="12:00"):
                        decision = conversation.classify(
                            text, "hours-variants", interaction_mode=InteractionMode.COMMAND
                        )
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

    def test_volume_commands_are_deterministic_and_never_call_llm(self) -> None:
        commands = {
            "volume em 0%": 0,
            "volume para 10%": 10,
            "coloque o volume em 50%": 50,
            "aumenta o volume para 80%": 80,
            "abaixa o volume para 100%": 100,
        }
        for text, percent in commands.items():
            with self.subTest(text=text):
                decision = self.conversation.classify(
                    f"Koda, {text}", "volume", interaction_mode=InteractionMode.WAKE
                )
                response = self.conversation.build_response_from_decision(decision)
                self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
                self.assertEqual(response["text"], f"Volume ajustado para {percent}%.")
                self.assertEqual(response["actions"], [{"type": "set_volume", "percent": percent}])
                self.assertEqual(extract_volume_percent(normalize_text(text)), percent)
        self.assertEqual(self.llm_client.calls, [])

    def test_volume_outside_supported_range_is_rejected_without_llm(self) -> None:
        for text in ("volume em 101%", "aumenta o volume para 255%", "volume para 1000%"):
            with self.subTest(text=text):
                decision = self.conversation.classify(
                    f"Koda, {text}", "invalid-volume", interaction_mode=InteractionMode.WAKE
                )
                response = self.conversation.build_response_from_decision(decision)
                self.assertEqual(response["text"], "O volume deve estar entre 0 e 100%.")
                self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm_client.calls, [])

    def test_llm_decision_is_executed_once_without_reclassification(self) -> None:
        self.conversation.classify("Koda", "one-llm", interaction_mode=InteractionMode.WAKE)
        with patch.object(self.conversation, "classify", wraps=self.conversation.classify) as classify:
            decision = self.conversation.classify(
                "Pergunta aberta", "one-llm", interaction_mode=InteractionMode.COMMAND
            )
            response = self.conversation.build_response_from_decision(decision)

        self.assertEqual(decision.kind, InteractionKind.LLM)
        self.assertEqual(classify.call_count, 1)
        self.assertEqual(len(self.llm_client.calls), 1)
        self.assertEqual(response["interaction_state"], "idle")

    def test_stream_llm_decision_is_executed_once_without_reclassification(self) -> None:
        llm_client = StreamingFakeLLMClient(("Resposta.",))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)
        conversation.classify("Koda", "one-stream", interaction_mode=InteractionMode.WAKE)
        with patch.object(conversation, "classify", wraps=conversation.classify) as classify:
            decision = conversation.classify(
                "Pergunta aberta", "one-stream", interaction_mode=InteractionMode.COMMAND
            )
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

        history = self.conversation.context_store._histories["rocky-01"].messages
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

        self.assertNotIn("rocky-01", self.conversation.context_store._histories)

    def test_hours_does_not_enter_history(self) -> None:
        self.wake()
        with patch("services.conversation.get_current_time", return_value="12:00"):
            self.conversation.build_response("horas", "rocky-01", interaction_mode=InteractionMode.COMMAND)

        self.assertNotIn("rocky-01", self.conversation.context_store._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_question_in_idle_does_not_enter_history(self) -> None:
        self.conversation.build_response("Quem criou o Linux?", "rocky-01")

        self.assertNotIn("rocky-01", self.conversation.context_store._histories)
        self.assertEqual(self.llm_client.calls, [])

    def test_empty_text_does_not_enter_history(self) -> None:
        self.wake()
        self.conversation.build_response("", "rocky-01", interaction_mode=InteractionMode.COMMAND)

        self.assertNotIn("rocky-01", self.conversation.context_store._histories)
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

        conversation.build_stream_response_from_decision(
            conversation.classify("Koda", "streaming-rocky", interaction_mode=InteractionMode.WAKE)
        )
        result = conversation.build_stream_response_from_decision(
            conversation.classify("Me conte algo", "streaming-rocky", interaction_mode=InteractionMode.COMMAND)
        )

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

                conversation.build_stream_response_from_decision(
                    conversation.classify(
                        "Koda", f"streaming-error-{status_code}", interaction_mode=InteractionMode.WAKE
                    )
                )
                result = conversation.build_stream_response_from_decision(
                    conversation.classify(
                        "Pergunta", f"streaming-error-{status_code}", interaction_mode=InteractionMode.COMMAND
                    )
                )

                self.assertEqual(len(llm_client.stream_calls), 1)
                self.assertEqual(result.sentences, ())
                self.assertEqual(result.response["text"], LLM_UNAVAILABLE_MESSAGE)

    def test_stream_error_after_partial_text_never_retries(self) -> None:
        llm_client = StreamingFakeLLMClient(("Resposta parcial"), LLMError("OpenRouter HTTP 429"))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)

        conversation.build_stream_response_from_decision(
            conversation.classify("Koda", "streaming-partial", interaction_mode=InteractionMode.WAKE)
        )
        result = conversation.build_stream_response_from_decision(
            conversation.classify("Pergunta", "streaming-partial", interaction_mode=InteractionMode.COMMAND)
        )

        self.assertEqual(len(llm_client.stream_calls), 1)
        self.assertEqual(result.sentences, ("Resposta parcial",))
        self.assertEqual(result.response["text"], f"Resposta parcial {LLM_INTERRUPTED_MESSAGE}")

    def test_streaming_hours_remains_deterministic_without_llm(self) -> None:
        llm_client = StreamingFakeLLMClient(("Não deve ser usado.",))
        conversation = ConversationManager(10000, llm_client, clock=self.clock)

        conversation.build_stream_response_from_decision(
            conversation.classify("Koda", "streaming-hours", interaction_mode=InteractionMode.WAKE)
        )
        with patch("services.conversation.get_current_time", return_value="12:00"):
            result = conversation.build_stream_response_from_decision(
                conversation.classify("horas", "streaming-hours", interaction_mode=InteractionMode.COMMAND)
            )

        self.assertEqual(llm_client.stream_calls, [])
        self.assertEqual(result.response["actions"][0]["type"], "show_text")


class FirmwareAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        verbose_patch = patch.dict(os.environ, {"KODA_VERBOSE": "1"})
        verbose_patch.start()
        self.addCleanup(verbose_patch.stop)
        self.clock = FakeClock()
        monotonic_patch = patch("services.conversation.time.monotonic", side_effect=self.clock)
        self.monotonic = monotonic_patch.start()
        self.addCleanup(monotonic_patch.stop)
        self.llm = FakeLLMClient()
        self.manager = ConversationManager(10000, self.llm, clock=self.monotonic)
        self.session = CommandSession()

    def wake(self, device_id: str = "koda-test") -> dict[str, object]:
        response = self.manager.build_response("Koda", device_id, interaction_mode=InteractionMode.WAKE)
        if response["interaction_state"] == "attention":
            self.session.open(device_id)
        return response

    def hours(self, received_at: float | None = None, device_id: str = "koda-test",
              mode: InteractionMode = InteractionMode.COMMAND):
        with patch("services.conversation.get_current_time", return_value="12:00"):
            decision = self.manager.classify("Que horas são?", device_id,
                                             received_at=received_at, interaction_mode=mode)
        response = self.manager.build_response_from_decision(decision)
        if mode == InteractionMode.COMMAND and response["interaction_state"] != "attention":
            self.session.close(device_id)
        return decision, response

    def assert_hours(self, decision, response) -> None:
        self.assertEqual(decision.kind, InteractionKind.DETERMINISTIC)
        self.assertEqual(response["actions"][0]["type"], "show_text")
        self.assertEqual(response["actions"][0]["line2"], "12:00")
        self.assertEqual(self.llm.calls, [])

    def test_wake_returns_attention_without_backend_deadline(self) -> None:
        response = self.wake()
        self.assertEqual(response["interaction_state"], "attention")
        self.assertNotIn("command_window_ms", response)
        self.assertFalse(hasattr(self.manager, "_deadlines"))
        self.assertTrue(self.session.is_active("koda-test"))

    def test_question_at_one_second_is_processed_without_timeout(self) -> None:
        self.wake()
        self.clock.now = 1.0
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertTrue(self.session.is_active("koda-test"))
            self.assert_hours(*self.hours())
        self.assertIn("state=waiting_command", output.getvalue())
        self.assertNotIn("timeout -> idle", output.getvalue())

    def test_capture_started_at_five_point_nine_is_valid_after_slow_stt(self) -> None:
        self.wake()
        self.clock.now = 5.9
        received_at = self.manager.capture_request_time()
        self.clock.now = 50.0
        self.assertTrue(self.session.is_active("koda-test"))
        self.assert_hours(*self.hours(received_at=received_at))

    def test_firmware_timeout_returns_to_wake_mode_and_ignores_question(self) -> None:
        self.wake()
        self.clock.now = 6.001
        output = io.StringIO()
        with redirect_stdout(output):
            decision, response = self.hours(mode=InteractionMode.WAKE)
        self.assertEqual(decision.kind, InteractionKind.EMPTY)
        self.assertEqual(response["interaction_state"], "idle")
        self.assertNotIn("timeout -> idle", output.getvalue())
        self.session.close("koda-test")
        self.assertFalse(self.session.is_active("koda-test"))
        self.assertEqual(self.llm.calls, [])

    def test_new_request_trackers_do_not_invalidate_command_mode(self) -> None:
        with latency_scope(clock=lambda: 50000.0):
            self.wake()
        self.clock.now = 1.0
        tracker = LatencyTrace(clock=lambda: 0.0)
        tracker.mark("stt_start")
        with latency_scope(clock=lambda: 999999.0):
            self.assertTrue(self.session.is_active("koda-test"))
            self.assert_hours(*self.hours())

    def test_slow_stt_does_not_expire_a_command_received_within_window(self) -> None:
        for arrival in (1.0, 5.9):
            with self.subTest(arrival=arrival):
                self.clock.now = 0.0
                self.wake()
                self.clock.now = arrival
                received_at = self.manager.capture_request_time()
                self.clock.now = 167.797
                self.assertTrue(self.session.is_active("koda-test"))
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assert_hours(*self.hours(received_at=received_at))
                self.assertNotIn("timeout -> idle", output.getvalue())

    def test_command_mode_needs_no_second_wake_and_no_backend_clock(self) -> None:
        self.wake()
        self.clock.now = 500.0
        with latency_scope(clock=lambda: 0.0):
            self.assert_hours(*self.hours())

    def test_command_windows_remain_separated_by_device_id(self) -> None:
        self.wake("koda-a")
        self.clock.now = 1.0
        _, idle = self.hours(device_id="koda-b", mode=InteractionMode.WAKE)
        self.assertEqual(idle["actions"], [])
        self.assertTrue(self.session.is_active("koda-a"))
        self.clock.now = 5.0
        self.wake("koda-b")
        self.clock.now = 11.0
        self.assertTrue(self.session.is_active("koda-a"))
        self.assertTrue(self.session.is_active("koda-b"))
        self.assert_hours(*self.hours(device_id="koda-b"))

    def test_explicit_wake_mode_does_not_reuse_a_stale_session(self) -> None:
        self.wake()
        response = self.manager.build_response("Quem criou o Linux?", "koda-test",
                                               interaction_mode=InteractionMode.WAKE)
        self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm.calls, [])

    def test_command_open_question_calls_llm_once(self) -> None:
        self.wake()
        self.clock.now = 60.0
        response = self.manager.build_response("Quem criou o Linux?", "koda-test",
                                               interaction_mode=InteractionMode.COMMAND)
        self.assertTrue(response["text"])
        self.assertEqual(len(self.llm.calls), 1)

    def test_command_mode_treats_koda_as_command_instead_of_reopening_attention(self) -> None:
        self.wake()
        response = self.manager.build_response("Koda", "koda-test", interaction_mode=InteractionMode.COMMAND)
        self.assertEqual(response["interaction_state"], "idle")
        self.assertEqual(len(self.llm.calls), 1)

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.manager.classify("Pergunta", "koda-test", interaction_mode="invalid")


if __name__ == "__main__":
    unittest.main()
