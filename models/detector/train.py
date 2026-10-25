"""Fine-tune bert-base-multilingual-cased for binary injection detection."""
from __future__ import annotations

import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.utils import get_logger, load_config
from src.utils.tracking import wandb_enabled

logger = get_logger("detector_train")

BASE_MODEL = "bert-base-multilingual-cased"
LABEL2ID = {"BENIGN": 0, "INJECTION": 1}
ID2LABEL = {0: "BENIGN", 1: "INJECTION"}


# Same on/off rule as the experiment scripts (src/utils/tracking.py).
_wandb_enabled = wandb_enabled


def _eval_strategy_key() -> str:
    """transformers >= 4.46 renamed `evaluation_strategy` -> `eval_strategy`."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


def _tokenizer_kwarg_key() -> str:
    """transformers >= 4.46 renamed the Trainer `tokenizer=` kwarg to
    `processing_class=` (and later versions drop `tokenizer` entirely)."""
    params = inspect.signature(Trainer.__init__).parameters
    return "processing_class" if "processing_class" in params else "tokenizer"


def _group_aware_split(
    rows: list[dict],
    val_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split by GROUP (``group_id``), not by row, so every augmented near-duplicate
    of a sentence lands in the same split as its original. Splitting flat rows
    (the old behaviour) leaks: an augmented variant of a test-set sentence can
    end up in train, which is why the first training run scored a suspicious
    eval_f1 = 1.0 — the model had effectively seen the test sentences already.
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)

    group_ids = list(groups)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n = len(group_ids)
    n_val = max(1, int(n * val_split))
    n_test = max(1, int(n * test_split))
    if n_val + n_test >= n:
        raise RuntimeError(
            f"Too few distinct base samples ({n} groups) for the requested "
            f"val/test splits — add more base samples or lower the split ratios."
        )

    val_ids = set(group_ids[:n_val])
    test_ids = set(group_ids[n_val:n_val + n_test])

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for gid, items in groups.items():
        bucket = val_rows if gid in val_ids else (test_rows if gid in test_ids else train_rows)
        bucket.extend(items)
    return train_rows, val_rows, test_rows


def load_dataset_from_json(
    benign_path: str,
    malicious_path: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
) -> DatasetDict:
    def _read(path: str, label: int) -> list[dict]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        out = []
        for item in data if isinstance(data, list) else []:
            text = item.get("payload", item.get("text", ""))
            if not text:
                continue
            # Older datasets (built before group-aware splitting existed) have
            # no group_id — fall back to a per-row id so they still work, just
            # without the leakage protection augmented data needs.
            group_id = item.get("group_id") or item.get("id") or text
            out.append({"text": text, "label": label, "group_id": group_id})
        return out

    benign = _read(benign_path, 0)
    malicious = _read(malicious_path, 1)
    all_data = benign + malicious
    if not all_data:
        raise RuntimeError(
            f"No training data found. Run scripts/build_detector_dataset.py first "
            f"(looked in {benign_path} and {malicious_path})."
        )

    train_rows, val_rows, test_rows = _group_aware_split(all_data, val_split, test_split, seed)
    n_groups = len({r["group_id"] for r in all_data})
    logger.info(
        "Dataset: %d rows in %d groups  ->  train=%d val=%d test=%d (group-aware split)",
        len(all_data), n_groups, len(train_rows), len(val_rows), len(test_rows),
    )

    def _strip(rows: list[dict]) -> list[dict]:
        return [{"text": r["text"], "label": r["label"]} for r in rows]

    return DatasetDict({
        "train": Dataset.from_list(_strip(train_rows)),
        "validation": Dataset.from_list(_strip(val_rows)),
        "test": Dataset.from_list(_strip(test_rows)),
    })


def load_ood_eval(path: str | Path) -> Dataset | None:
    """
    Load the hand-written, non-templated held-out set (built by
    scripts/build_detector_dataset.py). Returns None if it doesn't exist yet —
    training still proceeds, just without the OOD sanity check.
    """
    p = Path(path)
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = [
        {"text": item.get("payload", item.get("text", "")), "label": item["label"]}
        for item in data
        if item.get("payload") or item.get("text")
    ]
    return Dataset.from_list(rows) if rows else None


def tokenize_fn(examples, tokenizer, max_length: int = 512):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall": recall_score(labels, preds, average="binary", zero_division=0),
        "accuracy": float((preds == labels).mean()),
    }


def train(
    config_path: str = "configs/defense.yaml",
    use_wandb: bool = True,
    resume_from_checkpoint: str | None = None,
) -> dict:
    cfg = load_config(config_path)["detector"]
    td = cfg.get("training_data", {})
    wandb_on = _wandb_enabled(use_wandb)

    if wandb_on:
        import wandb

        wandb.init(project="prompt-injection-defense", name="detector-finetune", config=cfg)
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")
        logger.info("wandb logging disabled.")

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"],
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    dataset = load_dataset_from_json(
        td["benign"]["path"],
        td["malicious"]["path"],
        val_split=td.get("val_split", 0.15),
        test_split=td.get("test_split", 0.15),
    )
    tokenized = dataset.map(
        lambda x: tokenize_fn(x, tokenizer, cfg.get("max_length", 512)),
        batched=True,
        remove_columns=["text"],
    )

    strategy_kwargs = {_eval_strategy_key(): "epoch", "save_strategy": "epoch"}
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        load_best_model_at_end=cfg.get("save_best_only", True),
        # Each checkpoint is ~2 GB (weights + optimizer state); unbounded
        # saving filled the disk during a CPU run. The best checkpoint is
        # always kept on top of the most recent one.
        save_total_limit=cfg.get("save_total_limit", 2),
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=(["wandb"] if wandb_on else []),
        logging_steps=50,
        fp16=torch.cuda.is_available(),   # fp16 on CPU raises — gate on GPU
        **strategy_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=cfg["early_stopping_patience"]
        )],
        **{_tokenizer_kwarg_key(): tokenizer},
    )

    # ~4 min on a CUDA GPU (RTX 3060 laptop), ~2 h on CPU — install the CUDA
    # build of torch if a GPU is present. An interrupted run can continue from
    # its last checkpoint-N (the group-aware split is seeded, so the data is
    # identical).
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    best_dir = Path(cfg["output_dir"]) / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    test_results = trainer.evaluate(tokenized["test"])
    logger.info("In-distribution test results: %s", test_results)
    metrics_out = {"test": test_results}

    ood_path = Path(td["benign"]["path"]).parent / "ood_eval.json"
    ood_dataset = load_ood_eval(ood_path)
    if ood_dataset is not None:
        tokenized_ood = ood_dataset.map(
            lambda x: tokenize_fn(x, tokenizer, cfg.get("max_length", 512)),
            batched=True,
            remove_columns=["text"],
        )
        ood_results = trainer.evaluate(tokenized_ood, metric_key_prefix="ood")
        logger.info("OOD (held-out, non-templated) results: %s", ood_results)
        metrics_out["ood"] = ood_results

        test_f1 = test_results.get("eval_f1", 0.0)
        ood_f1 = ood_results.get("ood_f1", 0.0)
        if test_f1 - ood_f1 > 0.15:
            logger.warning(
                "In-distribution F1 (%.3f) is much higher than OOD F1 (%.3f) on "
                "hand-written, non-templated paraphrases — the detector likely "
                "learned surface patterns of the training templates rather than "
                "the underlying concept of injection. Treat the in-distribution "
                "score with caution; consider it the honest generalization number "
                "for the write-up.",
                test_f1, ood_f1,
            )
    else:
        logger.warning(
            "No OOD eval set found at %s — run scripts/build_detector_dataset.py "
            "to generate one (recommended before reporting eval_f1 in the write-up).",
            ood_path,
        )

    # Document-level OOD: held-out knowledge-base documents, clean vs. poisoned
    # with the Experience payloads (none of which appear in training). This is
    # the honest number for L2's retrieved-context scan.
    ood_doc_dataset = load_ood_eval(Path(td["benign"]["path"]).parent / "ood_doc_eval.json")
    if ood_doc_dataset is not None:
        tokenized_ood_doc = ood_doc_dataset.map(
            lambda x: tokenize_fn(x, tokenizer, cfg.get("max_length", 512)),
            batched=True,
            remove_columns=["text"],
        )
        ood_doc_results = trainer.evaluate(tokenized_ood_doc, metric_key_prefix="ood_doc")
        logger.info("OOD document-level results: %s", ood_doc_results)
        metrics_out["ood_doc"] = ood_doc_results

    (best_dir / "metrics.json").write_text(
        json.dumps(metrics_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if wandb_on:
        import wandb

        for split_name, split_metrics in metrics_out.items():
            wandb.log({f"{split_name}/{k}": v for k, v in split_metrics.items()})
        wandb.finish()

    return metrics_out


if __name__ == "__main__":
    train()
