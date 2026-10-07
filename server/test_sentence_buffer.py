"""Testes determinísticos do agrupamento local de sentenças em streaming."""

import unittest

from services.llm import SentenceBuffer


class SentenceBufferTest(unittest.TestCase):
    def test_decimal_thousands_and_abbreviations_are_not_boundaries(self) -> None:
        for text in ("Pi é 3.14 e custa 1.000 reais.", "O Sr. Silva chegou.",
                     "A Sra. Silva chegou.", "O Dr. Silva chegou.", "A Dra. Silva chegou."):
            with self.subTest(text=text):
                buffer = SentenceBuffer()
                self.assertEqual(buffer.push(text), (text,))

    def test_numeric_period_split_across_chunks(self) -> None:
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("Pi é 3."), ())
        self.assertEqual(buffer.push("14 e o Sr."), ())
        self.assertEqual(buffer.push(" Silva tem 1."), ())
        self.assertEqual(buffer.push("000 livros."), ("Pi é 3.14 e o Sr. Silva tem 1.000 livros.",))

    def test_numeric_sentence_waits_for_lookahead_or_finish(self) -> None:
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("Resposta: 42."), ())
        self.assertEqual(buffer.push(" Próxima."), ("Resposta: 42.", "Próxima."))
        self.assertEqual(buffer.push("42."), ())
        self.assertEqual(buffer.finish(), "42.")

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
