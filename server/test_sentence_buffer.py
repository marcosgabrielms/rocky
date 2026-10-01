"""Testes determinísticos do agrupamento local de sentenças em streaming."""

import unittest

from services.llm import SentenceBuffer


class SentenceBufferTest(unittest.TestCase):
    def test_emits_a_simple_complete_sentence(self) -> None:
        buffer = SentenceBuffer()

        self.assertEqual(buffer.push("Olá, Rocky."), ("Olá, Rocky.",))
        self.assertEqual(buffer.finish(), "")

    def test_joins_chunks_that_split_a_sentence(self) -> None:
        buffer = SentenceBuffer()

        self.assertEqual(buffer.push("Dois"), ())
        self.assertEqual(buffer.push(" mais dois são"), ())
        self.assertEqual(buffer.push(" quatro!"), ("Dois mais dois são quatro!",))

    def test_emits_multiple_sentences_from_one_or_more_chunks(self) -> None:
        buffer = SentenceBuffer()

        self.assertEqual(buffer.push("Primeira. Segunda?"), ("Primeira.", "Segunda?"))
        self.assertEqual(buffer.push(" Terceira!"), ("Terceira!",))

    def test_preserves_a_final_residual_without_forcing_punctuation(self) -> None:
        buffer = SentenceBuffer()

        self.assertEqual(buffer.push("Uma resposta ainda"), ())
        self.assertEqual(buffer.finish(), "Uma resposta ainda")

    def test_ignores_empty_and_whitespace_only_chunks(self) -> None:
        buffer = SentenceBuffer()

        self.assertEqual(buffer.push(""), ())
        self.assertEqual(buffer.push("   \n\t"), ())
        self.assertEqual(buffer.finish(), "")


if __name__ == "__main__":
    unittest.main()
