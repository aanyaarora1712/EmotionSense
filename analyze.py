"""Predict turn-level emotions and detect descriptive conversation patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from emotion_utils import EmotionPredictor


NEGATIVE = {"anger", "fear", "sadness"}
POSITIVE = {"joy", "love_caring"}


def summarize_conversation(group: pd.DataFrame) -> dict:
    group = group.sort_values("turn_id")
    emotions = group["emotion"].tolist()
    negative = [emotion in NEGATIVE for emotion in emotions]
    midpoint = max(1, len(emotions) // 2)
    early_negative = sum(negative[:midpoint]) / midpoint
    late_count = max(1, len(emotions) - midpoint)
    late_negative = sum(negative[midpoint:]) / late_count

    escalation = len(emotions) >= 3 and late_negative - early_negative >= 0.34
    recovery = any(
        emotions[i] in NEGATIVE and emotions[j] in POSITIVE
        for i in range(len(emotions))
        for j in range(i + 1, len(emotions))
    )
    repeated_frustration = sum(emotion == "anger" for emotion in emotions) >= 2
    dominant = pd.Series(emotions).mode().iloc[0]
    return {
        "conversation_id": group["conversation_id"].iloc[0],
        "user_turns": len(group),
        "dominant_emotion": dominant,
        "mean_confidence": round(group["confidence"].mean(), 4),
        "negative_escalation": escalation,
        "emotional_recovery": recovery,
        "repeated_frustration": repeated_frustration,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--include-all-speakers", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.input)
    required = {"conversation_id", "turn_id", "text"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "speaker" in data and not args.include_all_speakers:
        data = data[data["speaker"].str.lower() == "user"].copy()
    if data.empty:
        raise ValueError("No rows remain to analyze.")

    predictions = EmotionPredictor(args.model_dir).predict(data["text"].astype(str).tolist())
    data["emotion"] = [item["emotion"] for item in predictions]
    data["confidence"] = [item["confidence"] for item in predictions]
    data.to_csv(output_dir / "turn_predictions.csv", index=False)

    summaries = [summarize_conversation(group) for _, group in data.groupby("conversation_id")]
    pd.DataFrame(summaries).to_csv(output_dir / "conversation_patterns.csv", index=False)

    order = data["emotion"].value_counts().index
    plt.figure(figsize=(9, 5))
    sns.countplot(data=data, x="emotion", order=order, hue="emotion", legend=False)
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("Predicted emotion")
    plt.ylabel("Turns")
    plt.tight_layout()
    plt.savefig(output_dir / "emotion_distribution.png", dpi=180)
    plt.close()
    print(f"Analyzed {len(data)} turns across {data['conversation_id'].nunique()} conversations.")


if __name__ == "__main__":
    main()
