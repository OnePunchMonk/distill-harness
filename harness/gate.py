"""Escalation gate: decides whether a signal should trigger a teacher call.

Deliberately a plain threshold table, not a learned policy — the escalation
log is what tells you later whether thresholds are miscalibrated per the
design doc's open question on calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from harness.signals import Signal, SignalKind


@dataclass
class Budget:
    """Per-session / per-task-class escalation caps."""

    max_per_session: int
    max_per_task_class: dict[str, int] = field(default_factory=dict)
    _session_count: int = 0
    _class_counts: dict[str, int] = field(default_factory=dict)

    def has_room(self, task_class: str) -> bool:
        if self._session_count >= self.max_per_session:
            return False
        cap = self.max_per_task_class.get(task_class)
        if cap is not None and self._class_counts.get(task_class, 0) >= cap:
            return False
        return True

    def spend(self, task_class: str) -> None:
        self._session_count += 1
        self._class_counts[task_class] = self._class_counts.get(task_class, 0) + 1


@dataclass
class GateConfig:
    """Per-task-class thresholds. A missing task_class falls back to defaults."""

    default_confidence_threshold: float = 0.5
    escalate_on_structural_failure: bool = True
    escalate_on_sample_disagreement: bool = True
    per_class_confidence_threshold: dict[str, float] = field(default_factory=dict)


def should_escalate(
    signal: Signal,
    task_class: str,
    budget: Budget,
    config: GateConfig,
) -> bool:
    """Single policy function the harness calls between signal and teacher call."""
    if not budget.has_room(task_class):
        return False

    if signal.kind == SignalKind.STRUCTURAL:
        return config.escalate_on_structural_failure and signal.failed

    if signal.kind == SignalKind.SAMPLE_DISAGREEMENT:
        return config.escalate_on_sample_disagreement and signal.failed

    if signal.kind == SignalKind.SELF_REPORTED:
        threshold = config.per_class_confidence_threshold.get(
            task_class, config.default_confidence_threshold
        )
        return (signal.confidence or 0.0) < threshold

    return False
