"""SFT stage: fine-tune the student on distill/build.py's train.jsonl,
mixed with a replay slice of the original instruction-tuning data to guard
against regression. LoRA by default; fold to full fine-tune once a batch
proves itself on eval (see docs/design.md section 6).

This is a scaffold: the actual trainer wiring (transformers Trainer /
TRL SFTTrainer, tokenization, chat template application) is left as TODOs
so the model choice from configs/model.yaml can be plugged in without
rewriting this file's structure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SFTConfig:
    base_model: str
    train_path: str
    replay_path: str | None
    replay_frac: float
    output_dir: str
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 8


def load_mixed_dataset(cfg: SFTConfig) -> list[dict]:
    examples = [json.loads(l) for l in Path(cfg.train_path).read_text().splitlines() if l]

    if cfg.replay_path and Path(cfg.replay_path).exists():
        replay = [json.loads(l) for l in Path(cfg.replay_path).read_text().splitlines() if l]
        n_replay = int(len(examples) * cfg.replay_frac / (1 - cfg.replay_frac))
        examples.extend(replay[:n_replay])

    return examples


def run_sft(cfg: SFTConfig) -> str:
    """Returns the path to the resulting checkpoint.

    TODO: replace with a real trainer. Structure:
      1. load_mixed_dataset(cfg)
      2. apply the student's chat template to (prompt, chosen) pairs
      3. tokenize
      4. TRL SFTTrainer (or transformers Trainer) with LoRA config from cfg
      5. save adapter/checkpoint to cfg.output_dir
    """
    dataset = load_mixed_dataset(cfg)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"[sft] would train on {len(dataset)} examples -> {cfg.output_dir}")
    raise NotImplementedError("wire up TRL/transformers trainer here")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-path", default="data/build/train.jsonl")
    parser.add_argument("--replay-path", default=None)
    parser.add_argument("--replay-frac", type=float, default=0.2)
    parser.add_argument("--output-dir", default="data/checkpoints/sft")
    args = parser.parse_args()

    cfg = SFTConfig(
        base_model=args.base_model,
        train_path=args.train_path,
        replay_path=args.replay_path,
        replay_frac=args.replay_frac,
        output_dir=args.output_dir,
    )
    run_sft(cfg)
