"""Optional preference stage: DPO using (student attempt, teacher output)
as (rejected, chosen) pairs directly. Close to free since distill/build.py
already produces matched pairs from the escalation log. See docs/design.md
section 6, step 2.

Scaffold only — wire up TRL's DPOTrainer against the checkpoint produced
by posttrain/sft.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DPOConfig:
    base_checkpoint: str  # output of run_sft
    train_path: str
    output_dir: str
    beta: float = 0.1
    learning_rate: float = 5e-6
    epochs: int = 1
    batch_size: int = 4


def load_preference_pairs(train_path: str) -> list[dict]:
    examples = [json.loads(l) for l in Path(train_path).read_text().splitlines() if l]
    return [
        {"prompt": ex["prompt"], "chosen": ex["chosen"], "rejected": ex["rejected"]}
        for ex in examples
    ]


def run_dpo(cfg: DPOConfig) -> str:
    """Returns the path to the resulting checkpoint.

    TODO: wire up TRL DPOTrainer starting from cfg.base_checkpoint.
    """
    pairs = load_preference_pairs(cfg.train_path)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[dpo] would train on {len(pairs)} pairs -> {cfg.output_dir}")
    raise NotImplementedError("wire up TRL DPOTrainer here")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--train-path", default="data/build/train.jsonl")
    parser.add_argument("--output-dir", default="data/checkpoints/dpo")
    args = parser.parse_args()

    cfg = DPOConfig(
        base_checkpoint=args.base_checkpoint,
        train_path=args.train_path,
        output_dir=args.output_dir,
    )
    run_dpo(cfg)
