#!/usr/bin/env python3
"""Fine-tune an open-weight Llama-family model with LoRA on cite-or-refuse pairs.

Default runtime does not need this. CI runs ``--dry-run`` (dataset only).
GPU training: ``pip install -r requirements-lora.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset.jsonl"


def load_rows() -> list[dict]:
    """Read the JSONL cite-or-refuse dataset.

    Returns:
        One dict per non-empty line.
    """
    rows = []
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def validate(rows: list[dict]) -> None:
    """Check required fields and that answers are grounded or REFUSE.

    Args:
        rows: Dataset loaded by :func:`load_rows`.

    Raises:
        SystemExit: If the set is too small, a row is malformed, an answer has
            no context, or there are fewer than two REFUSE examples.
    """
    required = {"instruction", "context", "question", "output"}
    if len(rows) < 8:
        raise SystemExit("dataset too small")
    refuses = 0
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"row {i} missing {missing}")
        if not row["question"].strip() or not row["output"].strip():
            raise SystemExit(f"row {i} empty question/output")
        if row["output"].strip().upper() == "REFUSE":
            refuses += 1
        elif not row["context"].strip():
            raise SystemExit(f"row {i} answers without context")
    if refuses < 2:
        raise SystemExit("need refuse examples")


def format_prompt(row: dict) -> str:
    """Render one training example as a causal-LM string.

    Args:
        row: Keys ``instruction``, ``context``, ``question``, ``output``.

    Returns:
        Prompt ending in ``Answer: {output}``.
    """
    return (
        f"{row['instruction']}\n\n"
        f"Sources:\n{row['context'] or '(none)'}\n\n"
        f"Question: {row['question']}\n"
        f"Answer: {row['output']}"
    )


def train(args: argparse.Namespace, rows: list[dict]) -> None:
    """Run PEFT LoRA training and write the adapter.

    Args:
        args: Parsed CLI including ``base_model``, ``output_dir``, ``epochs``.
        rows: Validated dataset rows.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = [format_prompt(r) for r in rows]
    dataset = Dataset.from_dict({"text": texts})

    def tokenize(batch):
        """Pad and copy input ids to labels.

        Args:
            batch: Hugging Face batch with a ``text`` column.

        Returns:
            Tokenized batch including ``labels``.
        """
        out = tokenizer(batch["text"], truncation=True, max_length=512, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out),
            per_device_train_batch_size=1,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="epoch",
            report_to=[],
        ),
        train_dataset=tokenized,
    )
    trainer.train()
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"saved adapter to {out}")


def main() -> None:
    """Validate the dataset, then train unless ``--dry-run``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--output-dir", default=str(ROOT / "adapter"))
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    rows = load_rows()
    validate(rows)
    print(f"{len(rows)} cite-or-refuse examples ok")
    if args.dry_run:
        return
    train(args, rows)


if __name__ == "__main__":
    main()
