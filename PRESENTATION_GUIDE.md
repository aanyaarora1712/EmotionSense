# EmotionSense presentation guide

## What the system does

1. The UI accepts one user message per line.
2. A RoBERTa tokenizer converts each message into token IDs and an attention mask.
3. RoBERTa produces probabilities for the 28 original GoEmotions labels.
4. EmotionSense groups those labels into eight classes: anger, fear, joy,
   love/caring, sadness, surprise, neutral, and other.
5. Conversation logic examines the ordered predictions for dominant emotion,
   negative escalation, repeated frustration, and emotional recovery.
6. The UI presents the results and lets the presenter download them as CSV.

The demo automatically uses a public RoBERTa checkpoint trained on GoEmotions.
After a custom model is trained into `models/emotionsense`, the same UI detects
and loads it automatically.

## Local demo setup

Use Python 3.10 or 3.11. In the repository folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. The first analysis downloads
the model, so run it once before presenting. Later runs use the local cache.

## Test checklist

- Load **Support escalation** and confirm the result shows increasingly negative
  turns. The pattern flag needs at least three user turns.
- Load **Emotional recovery** and confirm a negative emotion is followed by a
  positive emotion.
- Load **Positive feedback** and confirm joy or love/caring dominates.
- Paste an ambiguous message such as `Fine.` and show the probability table to
  explain uncertainty.
- Download the CSV and confirm it contains text, emotion, and confidence.
- Refresh the page once before the presentation and keep a screenshot as backup.

Model outputs will not always match a human label. That is useful in the demo:
explain that context, sarcasm, short messages, class imbalance, and subjective
labels are known limitations.

## Recommended 3-minute demo

1. **Problem (20 seconds):** Support teams can read individual messages but have
   difficulty spotting emotional changes across long conversations.
2. **Architecture (30 seconds):** Explain input, tokenization, RoBERTa,
   eight-class mapping, and conversation-pattern logic.
3. **Live demo (90 seconds):** Run the escalation example, point to the
   turn-level table and pattern cards, then run the recovery example.
4. **Evaluation (25 seconds):** Show accuracy, macro-F1, and the confusion matrix
   from the custom training run. Macro-F1 matters because the classes are uneven.
5. **Limitations and next step (15 seconds):** The prototype analyzes language,
   not a person's mental state. The next step is evaluation on real, consented,
   domain-specific transcripts.

## One-week plan

- **Day 1:** Run the UI locally and test all three examples.
- **Day 2:** Run a quick custom training job in Google Colab.
- **Day 3:** Review `evaluation_metrics.json`, `confusion_matrix.png`, and the
  classification errors. Do not claim metrics until this run finishes.
- **Day 4:** Make 4–5 slides: problem, system flow, UI, results, limitations.
- **Day 5:** Deploy or record a backup screen capture.
- **Day 6:** Rehearse the three-minute script and test on the presentation Wi-Fi.
- **Day 7:** Freeze changes; keep the local model and screenshots available.
