"""Guard de fala deterministico: nao carrega Piper nem chama LLM."""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("piper", SimpleNamespace(PiperVoice=object))

from services.llm import SentenceBuffer
from services.tts.speech_text import SpeechTextSanitizer, log_speech_text


class SpeechTextTest(unittest.TestCase):
    def test_think_and_numeric_answer_regressions(self) -> None:
        sanitizer = SpeechTextSanitizer()
        for raw, expected in (
            ("<think>raciocinio interno</think> Paris.", "Paris."),
            ("<THINK>segredo. Outra frase!</THINK> Paris.", "Paris."),
            ("Paris. <think>segredo incompleto", "Paris."),
            ("Resposta: 42.", "42."), ("42.", "42."),
            ("1. Primeiro\n2. Segundo", "Primeiro Segundo"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(sanitizer.sanitize(raw), expected)

    def test_think_split_in_stream_never_reaches_speech(self) -> None:
        buffer = SentenceBuffer()
        sanitizer = SpeechTextSanitizer()
        spoken = []
        for chunk in ("<thi", "nk>segredo. ", "Outro segredo!", "</thi", "nk> Resposta: 42."):
            spoken.extend(sanitizer.sanitize(sentence) for sentence in buffer.push(chunk))
        spoken.append(sanitizer.sanitize(buffer.finish()))
        self.assertEqual([text for text in spoken if text], ["42."])

    def test_markdown_and_structural_labels(self) -> None:
        examples = (
            ("**Linus Torvalds** criou o Linux.", "Linus Torvalds criou o Linux."),
            ("*Linux* é um sistema.", "Linux é um sistema."),
            ("__Linux__", "Linux"), ("`Linux`", "Linux"),
            ("### Linux", "Linux"), ("- Linus Torvalds", "Linus Torvalds"),
            ("• Linus Torvalds", "Linus Torvalds"),
            ("[Linux](https://kernel.org)", "Linux"),
            ("https://kernel.org", ""),
            ("Input: Quem criou o Linux?", "Quem criou o Linux?"),
            ("Resposta: Linus Torvalds.", "Linus Torvalds."),
            ("Assistant: Linux. Output: Livre.", "Linux. Livre."),
            ("Analysis: Texto.\nAnswer: Resposta.", "Texto. Resposta."),
            ("Texto\n  Output: Resposta.", "Texto Resposta."),
            ("Input: Output: Linux.", "Linux."), ("1.", "1."),
            ("O input do programa está errado.", "O input do programa está errado."),
            ("A palavra Output: aparece nesta frase.", "A palavra Output: aparece nesta frase."),
            ("```python\nprint('codigo.')\n```\nResposta.", "Resposta."),
            ("```python\nprint('codigo.')", ""),
        )
        for raw, expected in examples:
            with self.subTest(raw=raw):
                clean = SpeechTextSanitizer().sanitize(raw)
                self.assertEqual(clean, expected)
                self.assertNotIn("*", clean)
                self.assertNotIn("`", clean)

    def test_does_not_summarize_or_truncate_sentences(self) -> None:
        text = " ".join(f"Frase completa numero {index}." for index in range(30))
        self.assertEqual(SpeechTextSanitizer().sanitize(text), text)

    def test_streaming_protects_link_and_code_until_cleanable(self) -> None:
        buffer = SentenceBuffer(preserve_line_breaks=True)
        sanitizer = SpeechTextSanitizer()
        spoken = []
        for chunk in ("Primeira. ", "[Linux](https://kernel.", "org) é livre. ",
                      "```python\nprint('Não fale.", "')\n```\nResposta: Segunda."):
            spoken.extend(sanitizer.sanitize(sentence) for sentence in buffer.push(chunk))
        residual = buffer.finish()
        if residual:
            spoken.append(sanitizer.sanitize(residual))
        self.assertEqual(spoken, ["Primeira.", "Linux é livre.", "Segunda."])

    def test_isolated_streamed_url_never_leaves_domain_fragments(self) -> None:
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("https://kernel."), ())
        self.assertEqual(buffer.push("org"), ())
        self.assertEqual(SpeechTextSanitizer().sanitize(buffer.finish()), "")

    def test_debug_logs_are_bounded_and_escape_line_breaks(self) -> None:
        output = io.StringIO()
        with patch.dict(os.environ, {"KODA_VERBOSE": "1"}), redirect_stdout(output):
            log_speech_text("LLM", "raw_sentence", 1, "\n" + "x" * 1000)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertLess(len(output.getvalue()), 320)


if __name__ == "__main__":
    unittest.main()
