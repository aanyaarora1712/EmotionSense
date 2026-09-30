"""Streamlit presentation UI for EmotionSense."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analyze import summarize_conversation
from emotion_utils import CLASS_NAMES, EmotionPredictor, GoEmotionsPredictor


EXAMPLES = {
    "Support escalation": [
        "I cannot find the setting I need.",
        "I already tried that twice and it still does not work.",
        "This is ridiculous. Why is it still broken?",
    ],
    "Emotional recovery": [
        "I am nervous that I entered the wrong information.",
        "Okay, I understand what happened now.",
        "That fixed it. Thank you so much!",
    ],
    "Positive feedback": [
        "This new feature is so easy to use.",
        "I love how quickly it found the answer.",
        "Amazing, that was exactly what I needed!",
    ],
}


st.set_page_config(page_title="EmotionSense", page_icon="ES", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem;}
    [data-testid="stMetric"] {background:#f6f3ee; border:1px solid #e7e0d7;
      padding:14px; border-radius:14px;}
    .eyebrow {letter-spacing:.12em; text-transform:uppercase; color:#7d7166;
      font-size:.78rem; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading RoBERTa model…")
def load_predictor():
    model_dir = Path("models/emotionsense")
    if (model_dir / "config.json").exists():
        return EmotionPredictor(model_dir), "Custom 8-class checkpoint"
    return GoEmotionsPredictor(), "Pretrained RoBERTa + GoEmotions"


def parse_turns(raw_text: str) -> list[str]:
    return [line.strip().lstrip("-• ").strip() for line in raw_text.splitlines() if line.strip()]


st.markdown('<div class="eyebrow">Conversation intelligence prototype</div>', unsafe_allow_html=True)
st.title("EmotionSense")
st.write(
    "Detect emotion in chatbot or voicebot transcripts and surface patterns such as "
    "escalation, repeated frustration, and emotional recovery."
)

with st.sidebar:
    st.header("Try a scenario")
    example_name = st.selectbox("Example", list(EXAMPLES))
    if st.button("Load example", use_container_width=True):
        st.session_state["conversation_input"] = "\n".join(EXAMPLES[example_name])
    st.divider()
    st.caption("Each line is treated as one user turn. Voicebot audio should be transcribed first.")

default_text = "\n".join(EXAMPLES["Support escalation"])
if "conversation_input" not in st.session_state:
    st.session_state["conversation_input"] = default_text
conversation = st.text_area(
    "Paste a conversation — one user message per line",
    height=180,
    key="conversation_input",
)

if st.button("Analyze conversation", type="primary", use_container_width=True):
    turns = parse_turns(conversation)
    if not turns:
        st.warning("Add at least one message to analyze.")
        st.stop()

    try:
        predictor, model_source = load_predictor()
        predictions = predictor.predict(turns)
    except Exception as exc:
        st.error("The model could not load. Check your internet connection and dependencies.")
        st.exception(exc)
        st.stop()

    rows = []
    for turn_id, (text, prediction) in enumerate(zip(turns, predictions), start=1):
        rows.append(
            {
                "conversation_id": "demo",
                "turn_id": turn_id,
                "text": text,
                "emotion": prediction["emotion"],
                "confidence": prediction["confidence"],
            }
        )
    frame = pd.DataFrame(rows)
    summary = summarize_conversation(frame)

    st.caption(f"Model: {model_source}")
    a, b, c, d = st.columns(4)
    a.metric("Dominant emotion", summary["dominant_emotion"].replace("_", " ").title())
    b.metric("Mean confidence", f'{summary["mean_confidence"]:.0%}')
    c.metric("Negative escalation", "Detected" if summary["negative_escalation"] else "Not detected")
    d.metric("Emotional recovery", "Detected" if summary["emotional_recovery"] else "Not detected")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Turn-by-turn analysis")
        display = frame[["turn_id", "text", "emotion", "confidence"]].copy()
        display.columns = ["Turn", "Message", "Emotion", "Confidence"]
        display["Emotion"] = display["Emotion"].str.replace("_", " ").str.title()
        display["Confidence"] = display["Confidence"].map(lambda value: f"{value:.0%}")
        st.dataframe(display, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Emotion distribution")
        counts = frame["emotion"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        st.bar_chart(counts, horizontal=True, color="#8f6f58")

    with st.expander("See class probabilities"):
        probability_rows = []
        for index, prediction in enumerate(predictions, start=1):
            probability_rows.append({"Turn": index, **prediction["probabilities"]})
        st.dataframe(pd.DataFrame(probability_rows), hide_index=True, use_container_width=True)

    st.download_button(
        "Download predictions as CSV",
        frame.to_csv(index=False),
        file_name="emotionsense_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Prototype only. Results describe language patterns and should not be used to infer "
    "mental-health conditions, intent, or identity."
)
