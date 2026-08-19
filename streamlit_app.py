"""
Streamlit demo UI for the Voice/Text Command Classifier.

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

st.set_page_config(page_title="Command Classifier", page_icon="📞", layout="centered")

# ---------------------------------------------------------------------------
# Landing page styling + hero section
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .hero {
        text-align: center;
        padding: 2.2rem 1rem 1.6rem 1rem;
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 18px;
        margin-bottom: 1.8rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.3rem;
        margin-bottom: 0.4rem;
        font-weight: 800;
    }
    .hero p {
        color: #d7e4ea;
        font-size: 1.05rem;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.5;
    }
    .feature-card {
        background: #ffffff08;
        border: 1px solid rgba(150,150,150,0.25);
        border-radius: 14px;
        padding: 1rem 1rem 0.9rem 1rem;
        height: 100%;
        transition: transform 0.15s ease;
    }
    .feature-card h4 {
        margin: 0 0 0.3rem 0;
        font-size: 1rem;
    }
    .feature-card p {
        margin: 0;
        font-size: 0.85rem;
        opacity: 0.85;
        line-height: 1.4;
    }
    .footnote {
        text-align: center;
        opacity: 0.55;
        font-size: 0.78rem;
        margin-top: 1.4rem;
        letter-spacing: 0.03em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📞 Talk to It. Don't Tap Through It.</h1>
    <p>An AI command classifier that understands what people actually say — typed or spoken,
    in plain English or Pidgin — and routes it instantly. No menus. No "press 1 for billing."</p>
</div>
""", unsafe_allow_html=True)

features = [
    ("⚡", "Instant Classification", "Type or speak a command, get a routed intent in real time."),
    ("🗣️", "Voice & Text, One Brain", "Whisper transcribes speech; the same model classifies both."),
    ("🇳🇬", "Pidgin-Aware", "Trained on Nigerian Pidgin phrasing, not just formal English."),
    ("🎯", "Knows What It Doesn't Know", "Low-confidence input is flagged, never silently misrouted."),
    ("🌍", "150-Intent General Mode", "Switch to a broad assistant model trained on real-world data."),
    ("📡", "Telecom-Ready", "22 purpose-built intents for balance, recharge, fraud, SIM issues."),
    ("🔍", "Transparent Confidence", "Every prediction ships with a full score breakdown."),
    ("🚀", "Built & Deployed", "Trained model, live API, working end-to-end — not just a slide."),
]

cols = st.columns(4)
for i, (icon, title, desc) in enumerate(features):
    with cols[i % 4]:
        st.markdown(f"""
        <div class="feature-card">
            <h4>{icon} {title}</h4>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    if i % 4 == 3 and i != len(features) - 1:
        cols = st.columns(4)

st.markdown('<div class="footnote">AI/ML Project</div>', unsafe_allow_html=True)
st.divider()

st.caption("Type a command or upload a short voice recording to see it classified.")

model_label = st.radio(
    "Choose a model",
    options=["telecom", "general"],
    format_func=lambda x: (
        "🇳🇬 Telecom IVR (Nigerian Pidgin-aware, 22 intents)"
        if x == "telecom"
        else "🌍 General assistant (150 intents)"
    ),
    horizontal=False,
)

tab_text, tab_voice, tab_intents = st.tabs(["💬 Text", "🎙️ Voice", "📋 Supported Intents"])


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
    st.subheader("Type a command")
    placeholder = (
        "e.g. 'abeg recharge my line with 500 naira' or 'block my sim I lost my phone'"
        if model_label == "telecom"
        else "e.g. 'what is the weather today' or 'set up direct deposit for my paycheck'"
    )
    text_input = st.text_input(placeholder, key="text_input")
    if st.button("Classify text", key="text_btn") and text_input.strip():
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
    st.subheader("Upload a voice recording")
    st.caption("Short audio clips work best (wav/mp3/m4a). Speech is transcribed with Whisper, then classified.")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"])
    if audio_file is not None and st.button("Classify audio", key="audio_btn"):
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
    st.subheader(f"Currently supported intents ({model_label})")
    try:
        resp = requests.get(f"{API_URL}/intents", params={"model": model_label}, timeout=5)
        resp.raise_for_status()
        intents = resp.json()["intents"]
        st.caption(f"{len(intents)} intents")
        for i in intents:
            st.markdown(f"- `{i}`")
    except Exception:
        st.info("Connect to the API to see the live intent list.")
