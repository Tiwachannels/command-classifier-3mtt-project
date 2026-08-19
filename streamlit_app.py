"""
Streamlit UI for the Voice/Text Command Classifier.

Run with: streamlit run streamlit_app.py
Expects the FastAPI backend to be running (default: http://localhost:8000).
Set API_URL via Streamlit secrets or the API_URL env var to point elsewhere.

Note on the floating input: st.chat_input always docks to the true bottom
of the browser viewport (that's a native Streamlit behavior, not CSS) so
the user never has to scroll to find it. It supports typed text AND
in-browser voice recording (accept_audio=True) in a single control.
"""
import os
import streamlit as st
import requests

try:
    API_URL = st.secrets["API_URL"]
except (KeyError, FileNotFoundError):
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Command Classifier", page_icon="📞", layout="wide")

# ---------------------------------------------------------------------------
# Global styling: Poppins font + light card backgrounds
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stMarkdown, .stButton button,
    .stTextInput input, .stRadio label, .stTabs [data-baseweb="tab"] {
        font-family: 'Poppins', sans-serif !important;
    }

    .model-card {
        background: #f4f4f6;
        border-radius: 14px;
        padding: 1.1rem 1rem;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .content-card {
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .result-empty {
        text-align: center;
        color: #9aa0a6;
        padding: 1.2rem 0;
        font-size: 0.92rem;
    }
    .model-card p {
    font-size: 1.2rem;
    }
    .model-card div[role="radiogroup"] {
        justify-content: center;
        gap: 1.5rem;
    }
    .model-card div[role="radiogroup"] label p {
        font-size: 1.05rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Topic groupings (sidebar)
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

PIDGIN_EXAMPLES = [
    "abeg check my balance for me",
    "block my sim, dem steal my phone",
    "abeg recharge my line with 500 naira",
]
ENGLISH_EXAMPLES = [
    "What's my current data balance?",
    "I'd like to activate roaming",
    "Can you connect me to an agent?",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "model_label" not in st.session_state:
    st.session_state.model_label = "telecom"


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def classify_text(text: str, model: str):
    try:
        resp = requests.post(
            f"{API_URL}/predict/text",
            json={"text": text, "model": model},
            timeout=10,
        )
        resp.raise_for_status()
        st.session_state.last_result = resp.json()
    except requests.exceptions.ConnectionError:
        st.session_state.last_result = {"error": f"Can't reach the API at {API_URL}. Is the backend running?"}
    except Exception as e:
        st.session_state.last_result = {"error": f"Error: {e}"}


def classify_audio(file_bytes: bytes, filename: str, model: str):
    try:
        files = {"file": (filename, file_bytes)}
        resp = requests.post(
            f"{API_URL}/predict/audio",
            params={"model": model},
            files=files,
            timeout=60,
        )
        resp.raise_for_status()
        st.session_state.last_result = resp.json()
    except requests.exceptions.ConnectionError:
        st.session_state.last_result = {"error": f"Can't reach the API at {API_URL}. Is the backend running?"}
    except Exception as e:
        st.session_state.last_result = {"error": f"Error: {e}"}


def render_result(result: dict):
    if result is None:
        st.markdown('<p class="result-empty">💡 Ask a question below, or tap a suggestion in the sidebar, '
                    'to see a prediction here.</p>', unsafe_allow_html=True)
        return
    if "error" in result:
        st.error(result["error"])
        return

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

    with st.expander("📋 Topics Covered", expanded=True):
        if st.session_state.model_label == "telecom":
            for category, items in TELECOM_TOPICS.items():
                st.markdown(f"**{category}**")
                for item in items:
                    st.markdown(f"- {item}")
        else:
            st.caption("150 intents grouped by category:")
            for category, count in GENERAL_TOPICS.items():
                st.markdown(
                    f"- {category}  \n  <span style='opacity:0.6;font-size:0.8em'>~{count} intents</span>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.markdown("#### 💡 Try asking")

    st.markdown("**🗣️ Pidgin**")
    for q in PIDGIN_EXAMPLES:
        if st.button(q, key=f"pg_{q}", use_container_width=True):
            classify_text(q, st.session_state.model_label)
            st.rerun()

    st.markdown("**🇬🇧 Plain English**")
    for q in ENGLISH_EXAMPLES:
        if st.button(q, key=f"en_{q}", use_container_width=True):
            classify_text(q, st.session_state.model_label)
            st.rerun()

# ---------------------------------------------------------------------------
# Main heading
# ---------------------------------------------------------------------------
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0.2rem;'>Command Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:gray; font-size:1.05rem;'>"
    "Say what you need — in English or Pidgin — and get routed instantly.</p>",
    unsafe_allow_html=True,
)
st.write("")

# ---------------------------------------------------------------------------
# Model selector card
# ---------------------------------------------------------------------------
st.markdown('<div class="model-card">', unsafe_allow_html=True)

st.markdown(
    "<p style='margin-bottom:0.4rem; font-weight:600; font-size:1.2rem;'>Model</p>",
    unsafe_allow_html=True
)

model_label = st.radio(
    "Model",
    options=["telecom", "general"],
    format_func=lambda x: (
        "📡 Telecom IVR" if x == "telecom"
        else "🌍 General Assistant"
    ),
    horizontal=True,
    label_visibility="collapsed",
    key="model_label",
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Content card: tabs for results / voice info / full intent list
# ---------------------------------------------------------------------------
st.markdown('<div class="content-card">', unsafe_allow_html=True)
tab_result, tab_voice, tab_intents = st.tabs(["💬 Result", "🎙️ Voice Tips", "📋 All Supported Intents"])

with tab_result:
    render_result(st.session_state.last_result)

with tab_voice:
    st.markdown(
        "Use the **🎤 record button** in the message bar at the bottom of the page to speak a command, "
        "Your speech is transcribed with Whisper, then classified the same way as typed text."
    )

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

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.8rem; margin-top:1.4rem;'>"
    "Voice/Text Command Classifier • AI/ML Project<br>"
    "Powered by TF-IDF + Logistic Regression, and OpenAI Whisper for speech</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Floating input, fixed to the bottom of the page (native st.chat_input
# behavior — no CSS positioning needed). Supports typed text AND in-browser
# voice recording in one control.
# ---------------------------------------------------------------------------
chat_value = st.chat_input(
    placeholder="Type a command, or tap the mic to speak it...",
    accept_audio=True,
)

if chat_value:
    if chat_value.audio is not None:
        classify_audio(chat_value.audio.getvalue(), "recording.wav", st.session_state.model_label)
    elif chat_value.text.strip():
        classify_text(chat_value.text, st.session_state.model_label)
    st.rerun()