"""
Streamlit UI for the Voice/Text Command Classifier.

Run with: streamlit run streamlit_app.py
Expects the FastAPI backend to be running (default: http://localhost:8000).
Set API_URL via Streamlit secrets or the API_URL env var to point elsewhere.
"""
import os
import streamlit as st
import requests

try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Voice/Text Command Classifier", page_icon="📞", layout="wide")

# ---------------------------------------------------------------------------
# Topic groupings (used in the sidebar). Telecom is fully enumerated since
# it's small (22 intents); general is grouped by rough category since 150
# individual intents would be overwhelming to list.
# ---------------------------------------------------------------------------
TELECOM_TOPICS = {
    "💰 Balance & Recharge": ["Check balance", "Recharge airtime", "Data balance", "Buy data bundle"],
    "📱 SIM & Security": ["Block SIM", "Unblock SIM", "Report fraud"],
    "📶 Network & Support": ["Report network issue", "Speak to agent", "General complaint"],
    "🎁 Plans & Offers": ["Change plan", "Check offers", "Activate roaming"],
    "💬 General": ["Call a contact", "Greetings", "Yes / No / Cancel / Repeat", "Thank you"],
}

GENERAL_TOPICS = {
    "🏦 Banking & Finance": 41,
    "✈️ Travel & Location": 23,
    "🏠 Vehicle & Home Utilities": 12,
    "🍳 Food & Recipes": 8,
    "🗓️ Productivity & Reminders": 17,
    "💬 Small Talk & Assistant Chat": 49,
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🇳🇬 Command Classifier")
    st.caption(
        "An AI system that understands spoken or typed commands — in plain "
        "English or Nigerian Pidgin — and routes them instantly."
    )
    st.divider()

    model_label = st.radio(
        "Model",
        options=["telecom", "general"],
        format_func=lambda x: "📡 Telecom IVR" if x == "telecom" else "🌍 General Assistant",
    )

    st.divider()

    with st.expander("📋 Topics Covered", expanded=True):
        if model_label == "telecom":
            for category, items in TELECOM_TOPICS.items():
                st.markdown(f"**{category}**")
                for item in items:
                    st.markdown(f"- {item}")
        else:
            st.caption("150 intents grouped by category:")
            for category, count in GENERAL_TOPICS.items():
                st.markdown(f"- {category}  \n  <span style='opacity:0.6;font-size:0.8em'>~{count} intents</span>",
                            unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main heading
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0.2rem;'>🇳🇬 Command Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray; font-size:1.05rem;'>"
    "Say what you need — in English or Pidgin — and get routed instantly.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Try asking
# ---------------------------------------------------------------------------
if "prefill_text" not in st.session_state:
    st.session_state.prefill_text = ""

st.markdown("#### 💡 Try asking")

pidgin_examples = [
    "abeg check my balance for me",
    "block my sim, dem steal my phone",
    "abeg recharge my line with 500 naira",
]
english_examples = [
    "What's my current data balance?",
    "I'd like to activate roaming",
    "Can you connect me to an agent?",
]

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🗣️ Pidgin**")
    for q in pidgin_examples:
        if st.button(q, key=f"pg_{q}", use_container_width=True):
            st.session_state.prefill_text = q
            st.rerun()
with col2:
    st.markdown("**🇬🇧 Plain English**")
    for q in english_examples:
        if st.button(q, key=f"en_{q}", use_container_width=True):
            st.session_state.prefill_text = q
            st.rerun()

st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem; margin-top:1rem;'>"
    "Voice/Text Command Classifier • AI/ML Project<br>"
    "Powered by TF-IDF + Logistic Regression, and OpenAI Whisper for speech</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Interaction area
# ---------------------------------------------------------------------------
tab_text, tab_voice, tab_intents = st.tabs(["💬 Text", "🎙️ Voice", "📋 All Supported Intents"])


def render_result(result: dict):
    intent = result["intent"]
    confidence = result["confidence"]
    transcript = result.get("transcript")

    if transcript:
        st.markdown(f"**Transcript:** _{transcript}_")

    if intent == "unrecognized":
        st.warning(f"⚠️ Not confident about this one (top guess confidence: {confidence:.0%}). "
                    f"Try rephrasing, or this may be an unsupported command.")
    else:
        st.success(f"**Predicted intent:** `{intent}`  \n**Confidence:** {confidence:.0%}")

    with st.expander("See all scores"):
        st.bar_chart(result["all_scores"])


with tab_text:
    text_input = st.text_input(
        "Ask a command...",
        value=st.session_state.prefill_text,
        key="text_input_field",
        label_visibility="collapsed",
        placeholder="e.g. 'abeg recharge my line with 500 naira' or 'check my balance'",
    )
    if st.button("Classify", key="text_btn", type="primary") and text_input.strip():
        try:
            resp = requests.post(
                f"{API_URL}/predict/text",
                json={"text": text_input, "model": model_label},
                timeout=10,
            )
            resp.raise_for_status()
            render_result(resp.json())
        except requests.exceptions.ConnectionError:
            st.error(f"Can't reach the API at {API_URL}. Is the backend running?")
        except Exception as e:
            st.error(f"Error: {e}")

with tab_voice:
    st.caption("Short audio clips work best (wav/mp3/m4a). Speech is transcribed with Whisper, then classified.")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"], label_visibility="collapsed")
    if audio_file is not None and st.button("Classify audio", key="audio_btn", type="primary"):
        with st.spinner("Transcribing and classifying..."):
            try:
                files = {"file": (audio_file.name, audio_file.getvalue())}
                resp = requests.post(
                    f"{API_URL}/predict/audio",
                    params={"model": model_label},
                    files=files,
                    timeout=60,
                )
                resp.raise_for_status()
                render_result(resp.json())
            except requests.exceptions.ConnectionError:
                st.error(f"Can't reach the API at {API_URL}. Is the backend running?")
            except Exception as e:
                st.error(f"Error: {e}")

with tab_intents:
    st.caption(f"Full intent list for the **{model_label}** model")
    try:
        resp = requests.get(f"{API_URL}/intents", params={"model": model_label}, timeout=5)
        resp.raise_for_status()
        intents = resp.json()["intents"]
        st.caption(f"{len(intents)} intents")
        cols = st.columns(3)
        for i, intent in enumerate(intents):
            cols[i % 3].markdown(f"- `{intent}`")
    except Exception:
        st.info("Connect to the API to see the live intent list.")
