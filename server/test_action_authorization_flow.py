"""Testes do fluxo conversacional de autorizacao de acoes futuras."""

import unittest

from koda_core import ActionPolicy, ConversationCoreAdapter, InteractionRequest, PendingActionAuthorizations
from services.action_authorization import ActionAuthorizationSession, AuthorizationStatus
from services.conversation import ConversationManager, InteractionMode
from services.llm import LLMResult


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _LLMDouble:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def ask(self, message: str, **_kwargs: object) -> LLMResult:
        self.calls.append(message)
        return LLMResult(text=f"Resposta: {message}")


class ActionAuthorizationFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = _Clock()
        self.llm = _LLMDouble()
        self.policy = ActionPolicy()
        self.pending = PendingActionAuthorizations(self.policy, identifier_factory=lambda: "authorization-1")
        self.session = ActionAuthorizationSession(
            self.policy,
            self.pending,
            clock=self.clock,
            ttl_seconds=120,
        )
        self.manager = ConversationManager(
            0,
            self.llm,
            clock=self.clock,
            action_authorization_session=self.session,
        )

    def _command(self, text: str, user_id: str = "local-user") -> dict[str, object]:
        return self.manager.build_response(
            text,
            "robot",
            user_id=user_id,
            interaction_mode=InteractionMode.COMMAND,
        )

    def test_allow_stays_direct_and_deny_never_creates_a_pending_authorization(self) -> None:
        allowed = self.manager.propose_action({"type": "set_volume", "percent": 50}, user_id="local-user")
        denied = self.manager.propose_action({"type": "purchase", "item": "produto"}, user_id="local-user")

        self.assertEqual(allowed["metadata"]["authorization"]["status"], "allowed")
        self.assertEqual(allowed["metadata"]["authorization"]["action"], {"type": "set_volume", "percent": 50})
        self.assertEqual(allowed["actions"], [])
        self.assertEqual(denied["metadata"]["authorization"]["status"], "denied")
        self.assertIsNone(self.session.get_pending("local-user"))

    def test_confirm_creates_one_pending_authorization_and_approval_returns_descriptor_without_execution(self) -> None:
        proposal = self.manager.propose_action(
            {"type": "send_message", "recipient": "contato", "text": "Ola"}, user_id="local-user",
        )
        repeated = self.manager.propose_action(
            {"type": "start_call", "recipient": "outro"}, user_id="local-user",
        )

        self.assertEqual(proposal["text"], "Posso fazer isso?")
        self.assertEqual(proposal["metadata"]["authorization"]["status"], "pending")
        self.assertEqual(repeated["metadata"]["authorization"]["status"], "pending")
        pending = self.session.get_pending("local-user")
        self.assertIsNotNone(pending)
        self.assertEqual(pending.action["type"], "send_message")

        response = self._command("pode fazer")

        self.assertEqual(response["metadata"]["authorization"]["status"], "approved")
        self.assertEqual(response["metadata"]["authorization"]["action"], pending.action)
        self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm.calls, [])
        self.assertIsNone(self.session.get_pending("local-user"))

    def test_clear_approval_and_rejection_phrases_are_deterministic_only_while_pending(self) -> None:
        for phrase in ("sim", "pode", "pode fazer", "autorizo", "confirma"):
            with self.subTest(phrase=phrase):
                self.manager.propose_action({"type": "modify_file", "path": "nota.txt"}, user_id="local-user")
                response = self._command(phrase)
                self.assertEqual(response["metadata"]["authorization"]["status"], "approved")

        for phrase in ("nao", "cancela", "nao faca"):
            with self.subTest(phrase=phrase):
                self.manager.propose_action({"type": "modify_file", "path": "nota.txt"}, user_id="local-user")
                response = self._command(phrase)
                self.assertEqual(response["metadata"]["authorization"]["status"], "rejected")

    def test_approval_cannot_be_reused_and_without_pending_follows_normal_conversation(self) -> None:
        self.manager.propose_action({"type": "change_configuration"}, user_id="local-user")
        self._command("sim")

        response = self._command("sim")

        self.assertEqual(response["text"], "Resposta: sim")
        self.assertNotIn("metadata", response)
        self.assertEqual(self.llm.calls, ["sim"])

    def test_expired_authorization_is_not_approved(self) -> None:
        self.manager.propose_action({"type": "run_task", "name": "backup"}, user_id="local-user")
        self.clock.now = 120.0

        response = self._command("sim")

        self.assertEqual(response["metadata"]["authorization"]["status"], "expired")
        self.assertEqual(response["actions"], [])
        self.assertEqual(self.llm.calls, [])
        self.assertIsNone(self.session.get_pending("local-user"))

    def test_authorization_is_shared_by_user_not_device_and_users_are_isolated(self) -> None:
        self.manager.propose_action({"type": "start_call", "recipient": "contato"}, user_id="same-user")

        other_user_response = self._command("sim", user_id="other-user")
        approved_response = self.manager.build_response(
            "sim",
            "android",
            user_id="same-user",
            interaction_mode=InteractionMode.COMMAND,
        )

        self.assertEqual(other_user_response["text"], "Resposta: sim")
        self.assertEqual(approved_response["metadata"]["authorization"]["status"], "approved")
        self.assertEqual(approved_response["metadata"]["authorization"]["action"]["type"], "start_call")

    def test_core_adapter_preserves_approved_descriptor_without_reclassification(self) -> None:
        adapter = ConversationCoreAdapter(self.manager)
        self.manager.propose_action({"type": "send_message", "text": "Ola"}, user_id="local-user")

        decision = adapter.classify(
            InteractionRequest("local-user", "robot", "text", "sim"),
            interaction_mode=InteractionMode.COMMAND,
        )
        result = adapter.build_direct_result(decision)

        self.assertEqual(result.metadata["authorization"]["status"], "approved")
        self.assertEqual(result.metadata["authorization"]["action"], {"type": "send_message", "text": "Ola"})
        self.assertEqual(self.llm.calls, [])


if __name__ == "__main__":
    unittest.main()
