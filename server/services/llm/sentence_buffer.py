"""Agrupa fragmentos de streaming em sentenças completas."""

import re


class SentenceBuffer:
    """Mantém somente o texto ainda não terminado por pontuação final."""

    _SENTENCE_ENDINGS = frozenset(".!?")
    _ABBREVIATIONS = frozenset({"sr", "sra", "dr", "dra"})
    _PROTECTED_TEXT = re.compile(
        r"(```|~~~).*?(?:\1|$)|<think\b[^>]*>.*?(?:</think\s*>|$)|!?\[[^\]]*\]\([^)]*(?:\)|$)|(?:https?://|www\.)\S+",
        re.DOTALL | re.IGNORECASE,
    )

    def __init__(self, preserve_line_breaks: bool = False) -> None:
        self._pending = ""
        self._preserve_line_breaks = preserve_line_breaks

    def push(self, chunk: str) -> tuple[str, ...]:
        """Adiciona um fragmento e devolve as sentenças que já terminaram."""
        if not chunk:
            return ()

        self._pending += chunk
        sentences: list[str] = []
        protected_spans = self._protected_spans()
        start = 0
        for index, character in enumerate(self._pending):
            if character not in self._SENTENCE_ENDINGS:
                continue
            if any(start <= index < end for start, end in protected_spans):
                continue
            if character == "." and self._is_nonterminal_period(index):
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

    def _is_nonterminal_period(self, index: int) -> bool:
        if index > 0 and self._pending[index - 1].isdigit():
            # Um ponto numerico no fim do chunk exige um caractere de lookahead.
            if index + 1 == len(self._pending) or self._pending[index + 1].isdigit():
                return True
        word = re.search(r"([A-Za-z]+)$", self._pending[:index])
        return word is not None and word.group(1).casefold() in self._ABBREVIATIONS

    def _protected_spans(self) -> tuple[tuple[int, int], ...]:
        spans = []
        for match in self._PROTECTED_TEXT.finditer(self._pending):
            end = match.end()
            if match.group().lower().startswith(("http://", "https://", "www.")) and end < len(self._pending):
                end -= len(match.group()) - len(match.group().rstrip(".!?"))
            spans.append((match.start(), end))
        return tuple(spans)

    def _normalize(self, text: str) -> str:
        normalized = re.sub(r"[^\S\n]+", " ", text).strip() if self._preserve_line_breaks else " ".join(text.split())
        return re.sub(r"\s+([,.;?!])", r"\1", normalized)
