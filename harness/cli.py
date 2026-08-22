"""Local smoke-test entrypoint: python -m harness.cli --input "..."

Uses stub models so it runs with no served checkpoint. Swap EchoStudent /
EchoTeacher for real backends once one is wired up.
"""

from __future__ import annotations

import argparse

from harness.gate import Budget, GateConfig
from harness.logging_ import EscalationLog
from harness.models import EchoStudent, EchoTeacher
from harness.runtime import Harness
from harness.signals import self_reported_signal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--task-class", default="default")
    parser.add_argument("--confidence", type=float, default=1.0,
                         help="fake self-reported confidence, for exercising the gate")
    parser.add_argument("--log-path", default="data/escalations.jsonl")
    args = parser.parse_args()

    harness = Harness(
        student=EchoStudent(),
        teacher=EchoTeacher(),
        log=EscalationLog(args.log_path),
        gate_config=GateConfig(default_confidence_threshold=0.5),
        budget=Budget(max_per_session=50),
    )

    result = harness.run(
        args.input,
        task_class=args.task_class,
        signal=self_reported_signal(args.confidence),
    )

    print(f"escalated: {result.escalated}")
    print(f"output: {result.output}")


if __name__ == "__main__":
    main()
