"""Testes da barreira HTTP do dispositivo Koda em modo SLEEPING."""

import unittest
from unittest.mock import patch

import app
from fastapi import HTTPException
from services.conversation import InteractionMode


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
        self.assertIsNone(app._sleeping_device_decision("Koda", self.device_id, InteractionMode.WAKE))
        bare_awake = app._sleeping_device_decision("acorda", self.device_id, InteractionMode.WAKE)
        self.assertEqual(bare_awake.kind, "empty")

        decision = app._sleeping_device_decision("acorda", self.device_id, InteractionMode.COMMAND)

        self.assertEqual(decision.kind, "deterministic")
        self.assertEqual(decision.direct_result.text, "Acordando.")
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
        classify.assert_not_called()

    def test_device_state_header_accepts_active_and_sleeping_only(self) -> None:
        self.assertFalse(app._is_sleeping_device_state("active"))
        self.assertTrue(app._is_sleeping_device_state("sleeping"))
        with self.assertRaises(HTTPException):
            app._is_sleeping_device_state("unknown")


if __name__ == "__main__":
    unittest.main()
