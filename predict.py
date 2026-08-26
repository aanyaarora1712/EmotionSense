"""Predict emotion for a single conversational message."""

import argparse
import json

from emotion_utils import EmotionPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    prediction = EmotionPredictor(args.model_dir).predict([args.text])[0]
    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
