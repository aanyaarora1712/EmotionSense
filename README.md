# EmotionSense: Emotion & Behavior Detection

EmotionSense fine-tunes `roberta-base` on the GoEmotions dataset and predicts eight
emotion classes in chatbot or voicebot transcripts:

`anger`, `fear`, `joy`, `love_caring`, `sadness`, `surprise`, `neutral`, `other`

It also analyzes conversation-level behavioral patterns such as negative-emotion
escalation, emotional recovery, repeated frustration, and dominant emotion.

## Run the presentation UI

The UI works immediately with a public RoBERTa model fine-tuned on GoEmotions.
If `models/emotionsense/config.json` exists, it automatically uses your custom
eight-class checkpoint instead.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, load an example or paste one user
message per line, and select **Analyze conversation**. The first run downloads
the pretrained model and can take a few minutes; later runs use the cache.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train and evaluate

For a quick local test:

```bash
python train.py --output-dir models/emotionsense --max-train-samples 5000 --epochs 1
```

For a full run:

```bash
python train.py --output-dir models/emotionsense --epochs 3
```

The training command saves the model, label mappings, evaluation metrics, a
confusion matrix, and a CSV containing misclassified validation examples.

## Predict one message

```bash
python predict.py --model-dir models/emotionsense --text "I have tried this three times and it still does not work."
```

## Analyze conversations

Input CSV columns:

- `conversation_id`: session identifier
- `turn_id`: numeric turn order
- `text`: user message or voice transcription
- `speaker`: optional; use `user` and `assistant`

```bash
python analyze.py \
  --model-dir models/emotionsense \
  --input data/sample_conversations.csv \
  --output-dir results
```

Outputs:

- `turn_predictions.csv`: class and confidence for every turn
- `conversation_patterns.csv`: dominant emotion, escalation, recovery, and
  repeated-frustration indicators for every conversation
- `emotion_distribution.png`: dataset-level class distribution

## Notes

- The default analysis predicts only rows whose `speaker` is `user`, because
  assistant language should not be treated as the user's emotion.
- Behavioral indicators are descriptive conversation patterns, not clinical or
  psychological diagnoses.
- Voicebot audio should first be transcribed to the same CSV schema.
