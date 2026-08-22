# distill-harness

A harness for a small open-weight student model that escalates to a larger
open-weight teacher model only when it needs to, and folds those escalations
back into the student via periodic post-training.

See `docs/design.md` for the full architecture writeup (student/teacher
selection, escalation gate, distillation build, post-training recipe, infra).

## Layout

```
harness/     runtime shell: student inference, escalation gate, teacher client, logging
distill/     turns the escalation log into a filtered/deduped/formatted train set
posttrain/   SFT + DPO training scripts, regression eval before promotion
eval/        held-out task-class eval sets and scoring
infra/       Modal app definitions (student serving, teacher fn, batch training job)
configs/     model, threshold, and training config yaml
data/        local escalation log + build artifacts (gitignored)
tests/       unit tests
```

## Status

Scaffold stage — interfaces and stubs are in place; model calls, real training
loops, and eval sets are not yet implemented.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m harness.cli --input "hello"
```
