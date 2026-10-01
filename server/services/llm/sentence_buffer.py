"""Agrupa fragmentos de streaming em sentenças completas."""

import re


class SentenceBuffer:
    """Mantém somente o texto ainda não terminado por pontuação final."""

    _SENTENCE_ENDINGS = frozenset(".!?")

    def __init__(self) -> None:
        self._pending = ""

    def push(self, chunk: str) -> tuple[str, ...]:
        """Adiciona um fragmento e devolve as sentenças que já terminaram."""
        if not chunk:
            return ()

        self._pending += chunk
        sentences: list[str] = []
        start = 0
        for index, character in enumerate(self._pending):
            if character not in self._SENTENCE_ENDINGS:
                continue

            sentence = self._normalize(self._pending[start : index + 1])
            if sentence:
                sentences.append(sentence)
            start = index + 1

        self._pending = self._pending[start:]
        return tuple(sentences)

    def finish(self) -> str:
        """Devolve o resíduo final, sem inventar uma pontuação inexistente."""
        residual = self._normalize(self._pending)
        self._pending = ""
        return residual

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = " ".join(text.split())
        return re.sub(r"\s+([,.;?!])", r"\1", normalized)
