"""Escalation log — the record that later becomes the distillation dataset.

One JSON line per escalation. Schema is a first-class artifact per the
design doc, not incidental telemetry — keep it stable, version it if it
must change.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EscalationRecord:
    schema_version: int
    timestamp: float
    task_class: str
    input: str
    student_attempt: str
    teacher_output: str
    verifier_verdict: str | None
    signal_kind: str


SCHEMA_VERSION = 1


class EscalationLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        task_class: str,
        input: str,
        student_attempt: str,
        teacher_output: str,
        signal_kind: str,
        verifier_verdict: str | None = None,
    ) -> EscalationRecord:
        record = EscalationRecord(
            schema_version=SCHEMA_VERSION,
            timestamp=time.time(),
            task_class=task_class,
            input=input,
            student_attempt=student_attempt,
            teacher_output=teacher_output,
            verifier_verdict=verifier_verdict,
            signal_kind=signal_kind,
        )
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        return record

    def read_all(self) -> list[EscalationRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(EscalationRecord(**json.loads(line)))
        return records
