"""Limpeza local de formatação para voz, sem resumir ou gerar conteúdo."""

import json
import re


SPEECH_TEXT_DIAGNOSTICS = True
MAX_LOG_CHARACTERS = 240


class SpeechTextSanitizer:
    def sanitize(self, text: str) -> str:
        text = re.sub(r"<think\b[^>]*>.*?(?:</think\s*>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(```|~~~).*?(?:\1|$)", "", text, flags=re.DOTALL)
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"(?:https?://|www\.)\S+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
        text = re.sub(r"(?m)^[ \t]*(?:#{1,6}\s*|[-+*•]\s+|\d+[.)][ \t]+(?=\S)|>\s+)", "", text)
        text = text.replace("*", "").replace("`", "").replace("__", "")
        text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
        text = re.sub(
            r"(^[ \t]*|(?<=[.!?])\s+)(?:(?:Input|Output|Assistant|Answer|Resposta|Analysis):\s*)+",
            r"\1", text, flags=re.IGNORECASE | re.MULTILINE,
        )
        text = re.sub(r"\s+([,.;?!])", r"\1", " ".join(text.split()))
        return text if any(character.isalnum() for character in text) else ""


def log_speech_text(source: str, stage: str, sequence: int, text: str) -> None:
    if SPEECH_TEXT_DIAGNOSTICS:
        preview = text[:MAX_LOG_CHARACTERS] + ("…" if len(text) > MAX_LOG_CHARACTERS else "")
        print(f"[{source}] {stage} seq={sequence} text={json.dumps(preview, ensure_ascii=False)}")
