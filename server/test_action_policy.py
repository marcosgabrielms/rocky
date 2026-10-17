"""Testes da politica de autorizacao pura para acoes futuras do Koda."""

import unittest

from koda_core import ActionPermission, ActionPolicy, PendingActionAuthorizations


class ActionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = ActionPolicy()

    def test_allowed_action_is_evaluated_deterministically(self) -> None:
        self.assertEqual(self.policy.evaluate({"type": "set_volume", "percent": 50}), ActionPermission.ALLOW)
        self.assertEqual(self.policy.evaluate({"type": "get_time"}), ActionPermission.ALLOW)

    def test_sensitive_action_requires_confirmation(self) -> None:
        self.assertEqual(
            self.policy.evaluate({"type": "send_message", "recipient": "contato", "text": "Ola"}),
            ActionPermission.CONFIRM,
        )
        self.assertEqual(self.policy.evaluate({"type": "future_unknown_action"}), ActionPermission.CONFIRM)

    def test_prohibited_or_malformed_action_is_denied(self) -> None:
        self.assertEqual(self.policy.evaluate({"type": "purchase"}), ActionPermission.DENY)
        self.assertEqual(self.policy.evaluate({"type": "access_bank"}), ActionPermission.DENY)
        self.assertEqual(self.policy.evaluate({"type": "bypass_biometric"}), ActionPermission.DENY)
        self.assertEqual(self.policy.evaluate({}), ActionPermission.DENY)

    def test_confirm_action_stays_pending_until_explicit_approval(self) -> None:
        authorizations = PendingActionAuthorizations(self.policy, identifier_factory=lambda: "confirm-1")
        proposed = authorizations.propose({"type": "modify_file", "path": "nota.txt"})

        self.assertEqual(proposed.permission, ActionPermission.CONFIRM)
        self.assertFalse(proposed.can_execute)
        self.assertEqual(authorizations.get_pending("confirm-1"), {"type": "modify_file", "path": "nota.txt"})
        self.assertEqual(authorizations.approve("confirm-1"), {"type": "modify_file", "path": "nota.txt"})
        self.assertIsNone(authorizations.get_pending("confirm-1"))
        self.assertIsNone(authorizations.approve("confirm-1"))

    def test_rejection_prevents_a_pending_action_from_being_released(self) -> None:
        authorizations = PendingActionAuthorizations(self.policy, identifier_factory=lambda: "confirm-2")
        authorizations.propose({"type": "start_call", "recipient": "contato"})

        self.assertTrue(authorizations.reject("confirm-2"))
        self.assertIsNone(authorizations.approve("confirm-2"))
        self.assertFalse(authorizations.reject("confirm-2"))

    def test_allowed_set_volume_remains_direct_and_policy_has_no_device_dependency(self) -> None:
        authorizations = PendingActionAuthorizations(self.policy)
        proposed = authorizations.propose({"type": "set_volume", "percent": 30})

        self.assertEqual(proposed.permission, ActionPermission.ALLOW)
        self.assertTrue(proposed.can_execute)
        self.assertIsNone(proposed.authorization_id)


if __name__ == "__main__":
    unittest.main()
