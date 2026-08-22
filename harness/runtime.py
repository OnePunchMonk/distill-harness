"""Runtime shell: routes a request through the student, decides whether to
escalate, and logs escalations for the distillation build.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.gate import Budget, GateConfig, should_escalate
from harness.logging_ import EscalationLog
from harness.models import ChatModel
from harness.signals import Signal, self_reported_signal


@dataclass
class HarnessResult:
    output: str
    escalated: bool
    task_class: str


class Harness:
    def __init__(
        self,
        student: ChatModel,
        teacher: ChatModel,
        log: EscalationLog,
        gate_config: GateConfig | None = None,
        budget: Budget | None = None,
    ):
        self.student = student
        self.teacher = teacher
        self.log = log
        self.gate_config = gate_config or GateConfig()
        self.budget = budget or Budget(max_per_session=50)

    def run(
        self,
        prompt: str,
        task_class: str = "default",
        signal: Signal | None = None,
    ) -> HarnessResult:
        student_output = self.student.generate(prompt)

        # Caller supplies a signal (structural check result, sample
        # disagreement, etc). Default to a neutral self-reported signal at
        # max confidence, i.e. "trust the student," when none is given.
        if signal is None:
            signal = self_reported_signal(confidence=1.0)

        if not should_escalate(signal, task_class, self.budget, self.gate_config):
            return HarnessResult(output=student_output, escalated=False, task_class=task_class)

        self.budget.spend(task_class)
        teacher_output = self.teacher.generate(prompt)
        self.log.write(
            task_class=task_class,
            input=prompt,
            student_attempt=student_output,
            teacher_output=teacher_output,
            signal_kind=signal.kind.value,
        )
        return HarnessResult(output=teacher_output, escalated=True, task_class=task_class)
