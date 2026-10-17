"""Testes da barreira HTTP do dispositivo Koda em modo SLEEPING."""

import unittest
from unittest.mock import patch

import app
from fastapi import HTTPException
from koda_core import CoreDecision, InteractionResult
from services.commands import DeviceStateIntent, detect_device_state_intent, normalize_device_state_text
from services.conversation import InteractionMode
from services.stt import TranscriptionResult


class DeviceSleepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device_id = "sleep-test-device"
        app.command_session.close(self.device_id)

    def tearDown(self) -> None:
        app.command_session.close(self.device_id)

    def test_sleeping_ignores_normal_speech_without_classifying_or_calling_llm(self) -> None:
        with patch.object(app.interaction_service, "classify") as classify:
            decision = app._classify_conversation(
                text="Koda, me conta uma piada",
                device_id=self.device_id,
                received_at=1.0,
                interaction_mode=InteractionMode.WAKE,
                low_confidence=False,
                device_sleeping=True,
            )

        self.assertEqual(decision.kind, "empty")
        self.assertEqual(decision.direct_result.text, "")
        self.assertEqual(decision.direct_result.actions, [])
        classify.assert_not_called()

    def test_sleeping_allows_wake_only_then_awake_command(self) -> None:
        self.assertIsNone(app._device_state_decision(
            "Koda", self.device_id, InteractionMode.WAKE, device_sleeping=True,
        ))
        bare_awake = app._device_state_decision(
            "acorda", self.device_id, InteractionMode.WAKE, device_sleeping=True,
        )
        self.assertIsNone(bare_awake)

        decision = app._device_state_decision(
            "acorda", self.device_id, InteractionMode.COMMAND, device_sleeping=True,
        )

        self.assertEqual(decision.kind, "deterministic")
        self.assertEqual(decision.direct_result.text, "Acordando.")
        self.assertEqual(decision.direct_result.interaction_state, "attention")
        self.assertEqual(
            decision.direct_result.actions,
            [{"type": "set_device_state", "state": "active"}],
        )

    def test_sleeping_accepts_koda_acorda_in_one_utterance_without_core(self) -> None:
        with patch.object(app.interaction_service, "classify") as classify:
            decision = app._classify_conversation(
                text="Koda, acorda",
                device_id=self.device_id,
                received_at=1.0,
                interaction_mode=InteractionMode.WAKE,
                low_confidence=False,
                device_sleeping=True,
            )

        self.assertEqual(decision.direct_result.actions[0]["state"], "active")
        self.assertEqual(decision.direct_result.interaction_state, "attention")
        self.assertTrue(app.command_session.is_active(self.device_id))
        classify.assert_not_called()

    def test_active_sleep_and_wake_device_commands_never_call_the_core(self) -> None:
        with patch.object(app.interaction_service, "classify") as classify:
            sleep_decision = app._classify_conversation(
                text="Koda, pode dormir",
                device_id=self.device_id,
                received_at=1.0,
                interaction_mode=InteractionMode.WAKE,
                low_confidence=False,
                device_sleeping=False,
            )
            wake_decision = app._classify_conversation(
                text="Koda, acorda",
                device_id=self.device_id,
                received_at=2.0,
                interaction_mode=InteractionMode.WAKE,
                low_confidence=False,
                device_sleeping=False,
            )

        self.assertEqual(sleep_decision.direct_result.actions, [{"type": "set_device_state", "state": "sleeping"}])
        self.assertEqual(wake_decision.direct_result.text, "Ja estou acordado.")
        self.assertEqual(wake_decision.direct_result.interaction_state, "attention")
        classify.assert_not_called()

    def test_active_command_window_recovers_observed_sleep_stt_errors_without_core(self) -> None:
        for text in ("vai dormir", "vai do b", "vai do rigido"):
            with self.subTest(text=text), patch.object(app.interaction_service, "classify") as classify:
                decision = app._classify_conversation(
                    text=text,
                    device_id=self.device_id,
                    received_at=1.0,
                    interaction_mode=InteractionMode.COMMAND,
                    low_confidence=False,
                    device_sleeping=False,
                )
            self.assertEqual(
                decision.direct_result.actions,
                [{"type": "set_device_state", "state": "sleeping"}],
            )
            classify.assert_not_called()

    def test_sleeping_command_window_recovers_cuda_as_wake_without_core(self) -> None:
        with patch.object(app.interaction_service, "classify") as classify:
            decision = app._classify_conversation(
                text="CUDA",
                device_id=self.device_id,
                received_at=1.0,
                interaction_mode=InteractionMode.COMMAND,
                low_confidence=False,
                device_sleeping=True,
            )

        self.assertEqual(decision.direct_result.actions, [{"type": "set_device_state", "state": "active"}])
        self.assertEqual(decision.direct_result.interaction_state, "attention")
        classify.assert_not_called()

    def test_ambiguous_short_post_wake_noise_stays_in_attention_without_core(self) -> None:
        for text in ("Bupi", "Hora de ouvir"):
            with self.subTest(text=text), patch.object(app.interaction_service, "classify") as classify:
                decision = app._classify_conversation(
                    text=text,
                    device_id=self.device_id,
                    received_at=1.0,
                    interaction_mode=InteractionMode.COMMAND,
                    low_confidence=False,
                    device_sleeping=False,
                )
            self.assertEqual(decision.direct_result.text, "")
            self.assertEqual(decision.direct_result.interaction_state, "attention")
            classify.assert_not_called()

    def test_sleeping_two_step_wake_then_question_reaches_normal_flow_once_awake(self) -> None:
        wake_core_decision = CoreDecision(
            kind="wake",
            text="Koda",
            device_id=self.device_id,
            direct_result=InteractionResult(text="", interaction_state="attention"),
        )
        question_core_decision = CoreDecision(
            kind="llm", text="me conta uma piada", device_id=self.device_id,
        )
        with patch.object(
            app.interaction_service,
            "classify",
            side_effect=(wake_core_decision, question_core_decision),
        ) as classify:
            wake_decision = app._classify_conversation(
                text="Koda",
                device_id=self.device_id,
                received_at=1.0,
                interaction_mode=InteractionMode.WAKE,
                low_confidence=False,
                device_sleeping=True,
            )
            awake_decision = app._classify_conversation(
                text="acorda",
                device_id=self.device_id,
                received_at=2.0,
                interaction_mode=None,
                low_confidence=False,
                device_sleeping=True,
            )
            question_decision = app._classify_conversation(
                text="me conta uma piada",
                device_id=self.device_id,
                received_at=3.0,
                interaction_mode=None,
                low_confidence=False,
                device_sleeping=False,
            )

        self.assertEqual(wake_decision.kind, "wake")
        self.assertEqual(awake_decision.direct_result.actions[0]["state"], "active")
        self.assertEqual(awake_decision.direct_result.interaction_state, "attention")
        self.assertEqual(question_decision.kind, "llm")
        self.assertEqual(classify.call_count, 2)
        self.assertEqual(classify.call_args_list[1].kwargs["interaction_mode"], InteractionMode.COMMAND)

    def test_sleeping_accepts_natural_wake_variants_without_calling_the_core(self) -> None:
        variants = (
            "Koda, acorda", "Koda acorda", "acorda, Koda", "Koda, pode acordar",
            "Koda, acorda aí", "Koda, você pode despertar aí?", "Koda, acora",
        )
        for text in variants:
            with self.subTest(text=text), patch.object(app.interaction_service, "classify") as classify:
                decision = app._classify_conversation(
                    text=text,
                    device_id=self.device_id,
                    received_at=1.0,
                    interaction_mode=InteractionMode.WAKE,
                    low_confidence=True,
                    device_sleeping=True,
                )
            self.assertEqual(decision.direct_result.interaction_state, "attention")
            self.assertEqual(decision.direct_result.actions[0]["state"], "active")
            classify.assert_not_called()

    def test_sleeping_accepts_bare_wake_only_inside_existing_command_window(self) -> None:
        decision = app._classify_conversation(
            text="acorda",
            device_id=self.device_id,
            received_at=1.0,
            interaction_mode=InteractionMode.COMMAND,
            low_confidence=True,
            device_sleeping=True,
        )

        self.assertEqual(decision.direct_result.actions[0]["state"], "active")

    def test_sleeping_preserves_low_confidence_text_only_for_strict_state_intent(self) -> None:
        transcription = TranscriptionResult(
            text="Koda, acorda", has_speech=True, low_confidence=True,
        )

        regular_text, regular_low_confidence = app._conversation_input(
            transcription, use_wake_hotword=True,
        )
        sleeping_text, sleeping_low_confidence = app._conversation_input(
            transcription,
            use_wake_hotword=True,
            allow_low_confidence_device_intent=True,
        )

        self.assertEqual(regular_text, "")
        self.assertTrue(regular_low_confidence)
        self.assertEqual(sleeping_text, "Koda, acorda")
        self.assertTrue(sleeping_low_confidence)

    def test_device_state_parser_rejects_common_phrases(self) -> None:
        for text in (
            "o que significa acordar?", "eu preciso dormir melhor", "fala sobre dormir",
        ):
            with self.subTest(text=text):
                self.assertIsNone(detect_device_state_intent(text, require_wake_word=True))

    def test_common_sleep_words_do_not_change_device_state_inside_command_window(self) -> None:
        for text in ("eu preciso dormir melhor", "fala sobre dormir"):
            with self.subTest(text=text):
                self.assertIsNone(app._device_state_decision(
                    text,
                    self.device_id,
                    InteractionMode.COMMAND,
                    device_sleeping=False,
                ))

    def test_device_state_parser_normalizes_accents_and_limits_fuzzy_matching(self) -> None:
        self.assertEqual(normalize_device_state_text("Koda, vá dormir!"), "koda va dormir")
        self.assertIs(
            detect_device_state_intent("Koda, durmir", require_wake_word=True),
            DeviceStateIntent.SLEEP_DEVICE,
        )
        self.assertIsNone(detect_device_state_intent("Koda, acordando", require_wake_word=True))

    def test_device_state_header_accepts_active_and_sleeping_only(self) -> None:
        self.assertFalse(app._is_sleeping_device_state("active"))
        self.assertTrue(app._is_sleeping_device_state("sleeping"))
        with self.assertRaises(HTTPException):
            app._is_sleeping_device_state("unknown")


if __name__ == "__main__":
    unittest.main()
