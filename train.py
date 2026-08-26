"""Fine-tune RoBERTa on an eight-class version of GoEmotions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from emotion_utils import CLASS_NAMES, compute_metrics, load_goemotions_8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="roberta-base")
    parser.add_argument("--output-dir", default="models/emotionsense")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_goemotions_8(args.seed)
    if args.max_train_samples:
        count = min(args.max_train_samples, len(data["train"]))
        data["train"] = data["train"].shuffle(seed=args.seed).select(range(count))
    if args.max_eval_samples:
        count = min(args.max_eval_samples, len(data["validation"]))
        data["validation"] = data["validation"].shuffle(seed=args.seed).select(range(count))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = data.map(tokenize, batched=True)
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(CLASS_NAMES),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    predictions = trainer.predict(tokenized["validation"])
    predicted_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = np.asarray(tokenized["validation"]["label"])
    matrix = confusion_matrix(true_ids, predicted_ids, labels=range(len(CLASS_NAMES)))

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    errors = pd.DataFrame(
        {
            "text": data["validation"]["text"],
            "actual": [id2label[i] for i in true_ids],
            "predicted": [id2label[i] for i in predicted_ids],
        }
    )
    errors = errors[errors["actual"] != errors["predicted"]]
    errors.to_csv(output_dir / "classification_errors.csv", index=False)
    clean_metrics = {key: float(value) for key, value in metrics.items()}
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(clean_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(clean_metrics, indent=2))


if __name__ == "__main__":
    main()
