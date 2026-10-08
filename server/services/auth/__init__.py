"""Autenticação local ChatGPT, independente do pipeline de inferência."""

from .chatgpt_session import ChatGPTSession
from .oauth import AuthError

__all__ = ["AuthError", "ChatGPTSession"]
