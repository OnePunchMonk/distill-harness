"""Regression gate: a new checkpoint only replaces the serving student if it
doesn't regress on held-out slices from every prior cycle, not just the
newest one. See docs/design.md section 6, step 3.

`score_fn` is a placeholder — plug in the real scorer (exact match, verifier
model, task-specific check) per task class.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

ScoreFn = Callable[[str, str], float]  # (model_output, reference) -> 0..1


def score_holdout(
    holdout_path: str | Path,
    generate: Callable[[str], str],
    score_fn: ScoreFn,
) -> dict[str, float]:
    """Average score per task class over one holdout file."""
    examples = [json.loads(l) for l in Path(holdout_path).read_text().splitlines() if l]
    totals: dict[str, list[float]] = {}
    for ex in examples:
        output = generate(ex["prompt"])
        score = score_fn(output, ex["chosen"])
        totals.setdefault(ex["task_class"], []).append(score)
    return {cls: sum(scores) / len(scores) for cls, scores in totals.items()}


def should_promote(
    new_scores: dict[str, float],
    baseline_scores: dict[str, float],
    tolerance: float = 0.02,
) -> tuple[bool, dict[str, float]]:
    """Compare new checkpoint's scores against the current serving model's
    scores on the same held-out classes. Promote only if nothing regresses
    beyond `tolerance`.
    """
    deltas = {}
    ok = True
    for cls, baseline in baseline_scores.items():
        new = new_scores.get(cls, 0.0)
        delta = new - baseline
        deltas[cls] = delta
        if delta < -tolerance:
            ok = False
    return ok, deltas
