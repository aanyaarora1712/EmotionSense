"""Shared label mapping, metrics, and inference helpers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CLASS_NAMES = [
    "anger",
    "fear",
    "joy",
    "love_caring",
    "sadness",
    "surprise",
    "neutral",
    "other",
]

GO_EMOTIONS_GROUPS = {
    "anger": {"anger", "annoyance", "disapproval", "disgust"},
    "fear": {"fear", "nervousness"},
    "joy": {"joy", "amusement", "excitement", "optimism", "pride", "relief"},
    "love_caring": {"love", "caring", "admiration", "approval", "gratitude", "desire"},
    "sadness": {"sadness", "disappointment", "grief", "remorse", "embarrassment"},
    "surprise": {"surprise", "realization", "confusion", "curiosity"},
    "neutral": {"neutral"},
}


def _fine_to_coarse(name: str) -> str:
    for coarse, fine_names in GO_EMOTIONS_GROUPS.items():
        if name in fine_names:
            return coarse
    return "other"


def load_goemotions_8(seed: int = 42) -> DatasetDict:
    """Load GoEmotions and convert its multi-label examples to one of 8 classes.

    Examples whose fine-grained labels map to multiple coarse classes are assigned
    to the least frequent coarse class among their labels. This deterministic
    choice retains ambiguous examples while reducing majority-class dominance.
    """
    raw = load_dataset("google-research-datasets/go_emotions", "simplified")
    fine_names = raw["train"].features["labels"].feature.names

    train_counts = Counter()
    for labels in raw["train"]["labels"]:
        train_counts.update({_fine_to_coarse(fine_names[i]) for i in labels})

    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    def convert(example):
        coarse = sorted({_fine_to_coarse(fine_names[i]) for i in example["labels"]})
        chosen = min(coarse, key=lambda name: (train_counts[name], CLASS_NAMES.index(name)))
        return {"label": class_to_id[chosen]}

    converted = DatasetDict()
    for split, dataset in raw.items():
        converted[split] = dataset.map(convert, remove_columns=["labels", "id"])
    return converted


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": macro_f1,
        "macro_precision": precision,
        "macro_recall": recall,
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
    }


class EmotionPredictor:
    def __init__(self, model_dir: str | Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.id_to_label = {
            int(key): value for key, value in self.model.config.id2label.items()
        }

    def predict(self, texts: list[str], batch_size: int = 32) -> list[dict]:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)
            confidence, prediction = probabilities.max(dim=-1)
            for pred, conf, probs in zip(prediction, confidence, probabilities):
                results.append(
                    {
                        "emotion": self.id_to_label[pred.item()],
                        "confidence": round(conf.item(), 4),
                        "probabilities": {
                            self.id_to_label[i]: round(value.item(), 4)
                            for i, value in enumerate(probs)
                        },
                    }
                )
        return results


class GoEmotionsPredictor:
    """Ready-to-demo predictor using a public RoBERTa GoEmotions checkpoint.

    The checkpoint predicts the original 28 GoEmotions labels. Probabilities are
    summed into this project's eight presentation-friendly classes, so the UI can
    run before a custom checkpoint has been trained.
    """

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    def predict(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=128, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                # This public checkpoint is multi-label, so each fine-grained
                # emotion receives an independent sigmoid probability.
                fine_probabilities = torch.sigmoid(self.model(**encoded).logits)

            for fine_probs in fine_probabilities:
                coarse = {name: 0.0 for name in CLASS_NAMES}
                for index, value in enumerate(fine_probs):
                    fine_name = self.model.config.id2label[index]
                    coarse[_fine_to_coarse(fine_name)] += value.item()
                total = sum(coarse.values()) or 1.0
                coarse = {name: value / total for name, value in coarse.items()}
                emotion = max(coarse, key=coarse.get)
                results.append(
                    {
                        "emotion": emotion,
                        "confidence": round(coarse[emotion], 4),
                        "probabilities": {
                            name: round(coarse[name], 4) for name in CLASS_NAMES
                        },
                    }
                )
        return results
