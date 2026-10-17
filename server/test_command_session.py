"""Testes do estado externo da janela de comando por dispositivo."""

import unittest

from services.command_session import CommandSession


class CommandSessionTest(unittest.TestCase):
    def test_open_close_and_device_isolation(self) -> None:
        session = CommandSession()

        self.assertFalse(session.is_active("koda-a"))
        session.open("koda-a")
        self.assertTrue(session.is_active("koda-a"))
        self.assertFalse(session.is_active("koda-b"))

        session.open("koda-b")
        session.close("koda-a")
        self.assertFalse(session.is_active("koda-a"))
        self.assertTrue(session.is_active("koda-b"))

    def test_close_is_idempotent(self) -> None:
        session = CommandSession()
        session.close("koda-a")
        session.open("koda-a")
        session.close("koda-a")
        session.close("koda-a")

        self.assertFalse(session.is_active("koda-a"))


if __name__ == "__main__":
    unittest.main()
