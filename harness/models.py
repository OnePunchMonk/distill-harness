"""Model client interfaces. Real backends (vLLM, Modal function, HF endpoint)
plug in behind these — the harness only depends on this protocol.
"""

from __future__ import annotations

from typing import Protocol


class ChatModel(Protocol):
    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Return a single completion for the prompt."""
        ...


class EchoStudent:
    """Placeholder student for local testing without a served model."""

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        return f"[student:stub] {prompt}"


class EchoTeacher:
    """Placeholder teacher for local testing without a served model."""

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        return f"[teacher:stub] {prompt}"
