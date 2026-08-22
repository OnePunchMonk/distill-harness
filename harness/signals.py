"""Confidence / failure signals used to decide whether to escalate to the teacher."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SignalKind(str, Enum):
    STRUCTURAL = "structural"       # tool call, schema, or code check failed
    SELF_REPORTED = "self_reported"  # student emitted a low-confidence flag
    SAMPLE_DISAGREEMENT = "sample_disagreement"  # two samples diverged


@dataclass
class Signal:
    kind: SignalKind
    failed: bool
    confidence: float | None = None  # 0-1, only set for self_reported
    detail: str = ""


def structural_signal(passed: bool, detail: str = "") -> Signal:
    """Task had a checkable output (schema/tool/code) that did or didn't validate."""
    return Signal(kind=SignalKind.STRUCTURAL, failed=not passed, detail=detail)


def self_reported_signal(confidence: float, threshold: float = 0.5) -> Signal:
    """Student emitted an explicit confidence score as part of its output format."""
    return Signal(
        kind=SignalKind.SELF_REPORTED,
        failed=confidence < threshold,
        confidence=confidence,
    )


def sample_disagreement_signal(sample_a: str, sample_b: str) -> Signal:
    """Two samples at moderate temperature; naive exact-match check.

    Replace with an embedding-similarity or task-specific equivalence check
    once the harness has a real similarity backend wired up.
    """
    disagreed = sample_a.strip() != sample_b.strip()
    return Signal(kind=SignalKind.SAMPLE_DISAGREEMENT, failed=disagreed)
