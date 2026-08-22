# On-demand distillation harness — design

## Objective

Post-train a small open-weight student, wrap it in a task harness, and call a
larger open-weight teacher model only when the student's own outputs signal
it needs one — turning distillation from a one-time offline pass into a
standing capability of the system.

"Distillation as needed" = teacher inference is gated behind an uncertainty
or failure signal from the student, not called unconditionally, and every
gated call is a labeled training example for the next post-training round.

## System shape

Three loops at different speeds:

- **Inference loop** (student, per request) — the harness's default path.
- **Escalation loop** (teacher, only on gated failures) — triggered by the
  escalation gate, logged in full.
- **Post-training loop** (batched, e.g. weekly or every N escalations) —
  consumes the escalation log, builds a train set, fine-tunes, evals,
  promotes a checkpoint.

```
input -> student -> [confident] -> output
                 \-> [escalate] -> teacher -> output
                                        \-> escalation log -> distill build -> post-train -> new student checkpoint
```

## Model selection

| Role | Candidate | Why |
|---|---|---|
| Student | Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct | strong instruction-following, cheap to serve |
| Teacher | Qwen2.5-72B-Instruct or Llama-3.1-70B-Instruct | same family as student — less format/style mismatch |
| Verifier (optional) | small task classifier, or teacher-as-judge | grades student attempts, informs escalation |

Same-family student/teacher isn't required but is the default: aligned chat
templates and tokenizers mean less reformatting before a teacher output is
usable as an SFT target.

## The harness

1. **Confidence / failure signal** — structural (tool/schema/code failure,
   no self-assessment needed), self-reported (confidence token in the
   output format), or sampling disagreement (two samples at moderate temp).
2. **Escalation gate** — `escalate(signal, task_class, budget_remaining) -> bool`.
   Start as a plain threshold table; let the log tell you when it's
   miscalibrated.
3. **Logging** — one record per escalation: input, student attempt (even if
   wrong), teacher output, verifier verdict, task class, timestamp. This log
   *is* the distillation dataset.
4. **Budget control** — cap escalations per session/task class; a task class
   escalating constantly is itself the strongest signal for what to
   prioritize next training cycle.

## Distillation build

Filter (drop teacher-also-failed or near-identical pairs) -> dedupe
(embedding similarity, against train set and within batch) -> rebalance
(cap per task class) -> format (recast into student's chat template) ->
hold out (per-class eval slice, never trained on).

Watch for drift toward the teacher's failure modes/style (verbosity,
hedging, refusal patterns) if filtering is too permissive.

## Post-training recipe

1. SFT on the filtered escalation set, mixed with a replay slice of the
   original instruction-tuning data to prevent regression. LoRA for fast
   iteration; fold to full fine-tune once a batch proves itself on eval.
2. Optional DPO/ORPO using (student attempt, teacher output) as
   (rejected, chosen) pairs directly — close to free since the log already
   contains matched pairs.
3. Regression eval against held-out slices from *every* prior cycle before
   promoting a checkpoint, not just the newest one.

## Infra

Modal-based: student serving stays warm (small, always-on endpoint), teacher
is a scale-to-zero function invoked only on escalation, post-training runs
as a scheduled batch job pulling the log from a shared volume.

## Open questions

- Escalation threshold calibration — start conservative or loose, and how
  fast does the verifier let you tighten it?
- Cycle cadence — fixed schedule vs. escalation-count trigger per class?
- Teacher cost ceiling — what actually bounds escalation rate / model size?
- Eval ownership — who defines task classes and revisits held-out sets as
  student capability shifts?
