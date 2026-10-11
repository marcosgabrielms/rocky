"""Testes determinísticos da configuração de logs do backend."""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from services.runtime_log import is_verbose, technical


class RuntimeLogTest(unittest.TestCase):
    def test_false_values_disable_verbose(self) -> None:
        for value in (None, "", "0", "false", "FALSE", "no", "off"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {} if value is None else {"KODA_VERBOSE": value},
                clear=value is None,
            ):
                self.assertFalse(is_verbose())

    def test_true_values_enable_verbose(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with self.subTest(value=value), patch.dict(os.environ, {"KODA_VERBOSE": value}):
                self.assertTrue(is_verbose())

    def test_technical_log_depends_on_runtime_value(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"KODA_VERBOSE": "0"}), redirect_stdout(output):
            technical("[RTD] hidden")
        self.assertEqual(output.getvalue(), "")

        with patch.dict(os.environ, {"KODA_VERBOSE": "1"}), redirect_stdout(output):
            technical("[RTD] visible")
        self.assertIn("[RTD] visible", output.getvalue())


if __name__ == "__main__":
    unittest.main()
