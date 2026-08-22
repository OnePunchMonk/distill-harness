"""Turns the escalation log into a training set: filter -> dedupe ->
rebalance -> format -> hold out. See docs/design.md section 5.

Dedup here is a stand-in (exact-match on normalized text). Swap in an
embedding-similarity backend before running this on a real log; a
production version should not train on filtered near-duplicates cheaply
skipped by string equality alone.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from harness.logging_ import EscalationRecord


@dataclass
class TrainExample:
    task_class: str
    prompt: str
    chosen: str  # teacher output, reformatted for the student's template
    rejected: str  # student's own attempt, for optional DPO use


def filter_records(records: list[EscalationRecord]) -> list[EscalationRecord]:
    """Drop records where teacher output is empty, or matches the student's
    attempt closely enough that there's nothing to learn from the pair.
    """
    out = []
    for r in records:
        if not r.teacher_output.strip():
            continue
        if r.verifier_verdict == "teacher_failed":
            continue
        if r.teacher_output.strip() == r.student_attempt.strip():
            continue
        out.append(r)
    return out


def dedupe_records(records: list[EscalationRecord]) -> list[EscalationRecord]:
    seen: set[str] = set()
    out = []
    for r in records:
        key = " ".join(r.input.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def rebalance_records(
    records: list[EscalationRecord], max_per_class: int
) -> list[EscalationRecord]:
    by_class: dict[str, list[EscalationRecord]] = defaultdict(list)
    for r in records:
        by_class[r.task_class].append(r)
    out = []
    for cls, group in by_class.items():
        random.shuffle(group)
        out.extend(group[:max_per_class])
    return out


def format_examples(records: list[EscalationRecord]) -> list[TrainExample]:
    return [
        TrainExample(
            task_class=r.task_class,
            prompt=r.input,
            chosen=r.teacher_output,
            rejected=r.student_attempt,
        )
        for r in records
    ]


def split_holdout(
    examples: list[TrainExample], holdout_frac: float = 0.1
) -> tuple[list[TrainExample], list[TrainExample]]:
    by_class: dict[str, list[TrainExample]] = defaultdict(list)
    for ex in examples:
        by_class[ex.task_class].append(ex)

    train, holdout = [], []
    for cls, group in by_class.items():
        random.shuffle(group)
        n_holdout = max(1, int(len(group) * holdout_frac)) if group else 0
        holdout.extend(group[:n_holdout])
        train.extend(group[n_holdout:])
    return train, holdout


def build(
    log_path: str | Path,
    out_dir: str | Path,
    max_per_class: int = 500,
    holdout_frac: float = 0.1,
) -> dict[str, int]:
    from harness.logging_ import EscalationLog

    records = EscalationLog(log_path).read_all()
    records = filter_records(records)
    records = dedupe_records(records)
    records = rebalance_records(records, max_per_class)
    examples = format_examples(records)
    train, holdout = split_holdout(examples, holdout_frac)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train)
    _write_jsonl(out_dir / "holdout.jsonl", holdout)

    return {"train": len(train), "holdout": len(holdout)}


def _write_jsonl(path: Path, examples: list[TrainExample]) -> None:
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="data/escalations.jsonl")
    parser.add_argument("--out-dir", default="data/build")
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--holdout-frac", type=float, default=0.1)
    args = parser.parse_args()

    counts = build(args.log_path, args.out_dir, args.max_per_class, args.holdout_frac)
    print(counts)
