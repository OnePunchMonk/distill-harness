"""Modal app skeleton: student stays warm, teacher scales to zero, batch
training job runs on a schedule. See docs/design.md section 7.

Stub — no real model loading yet. Fill in `student_generate` /
`teacher_generate` with vLLM or transformers inference once a GPU image is
picked, and point the volume at the shared escalation log.
"""

from __future__ import annotations

import modal

app = modal.App("distill-harness")

volume = modal.Volume.from_name("distill-harness-data", create_if_missing=True)

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
)


@app.function(image=image, gpu="A10G", min_containers=1, volumes={"/data": volume})
def student_generate(prompt: str) -> str:
    raise NotImplementedError("load student checkpoint from /data and generate")


@app.function(image=image, gpu="A100", scaledown_window=60, volumes={"/data": volume})
def teacher_generate(prompt: str) -> str:
    raise NotImplementedError("load/call teacher model and generate")


@app.function(image=image, gpu="A100", timeout=60 * 60 * 6, volumes={"/data": volume})
def run_posttraining_cycle() -> None:
    """Pulls the escalation log from the volume, builds the train set,
    runs SFT (+ optional DPO), evals against held-out slices, and promotes
    a checkpoint if it doesn't regress. See distill/build.py,
    posttrain/sft.py, posttrain/dpo.py, eval/promote.py.
    """
    raise NotImplementedError("wire up distill.build -> posttrain.sft/dpo -> eval.promote")


@app.function(schedule=modal.Period(days=7))
def scheduled_posttraining() -> None:
    run_posttraining_cycle.remote()
