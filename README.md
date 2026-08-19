## Voice/Text Command Classifier

## Experience the live demo version of this project at the following link
## https://voice-text-command-classifier.streamlit.app/
Classifies spoken or typed commands into intents. Ships with **two models**:

1. **`telecom`** — a Nigerian Pidgin-aware telecom IVR classifier (22 intents: balance
   checks, recharges, SIM blocking, fraud reports, etc.)
2. **`general`** — a 150-intent general-purpose assistant classifier (banking, travel,
   smart home, small talk) trained on a real-world dataset.

## Architecture

```
User (text or audio)
        |
        v
Streamlit UI (streamlit_app.py)  -- model selector: telecom / general
        |
        v
FastAPI backend (api/main.py)
   |-- /predict/text   --> app/model.py  (TF-IDF + Logistic Regression)
   |-- /predict/audio  --> app/stt.py (Whisper) --> app/model.py
        |
        v
   models/telecom_pipeline.joblib   or   models/full_150_pipeline.joblib
```

- **Text path:** text -> TF-IDF vectorizer -> Logistic Regression -> intent + confidence
- **Voice path:** audio file -> Whisper transcription -> same text pipeline
- **Fallback:** if the top prediction's confidence is below 0.35, the API returns
  `"unrecognized"` instead of forcing a guess.

## Data provenance (important)

### General model (150 intents)
Trained on a real-world English intent-classification dataset (CLINC150-style;
originally provided as English + French pairs — French was dropped as out of scope
for this project's Nigerian-market focus). 14,998 examples, 150 balanced intents.
This is real, naturally-phrased data, not synthetic.

### Telecom model (22 intents)
A blend of two sources:
- **Synthetic Pidgin/English telecom commands** (880 examples, template-generated)
  covering Nigeria-specific actions the general dataset doesn't have — airtime
  recharge, data bundles, SIM blocking, roaming activation, network complaints.
  This is what gives the model its local-context uniqueness, but it means these
  specific intents (`activate_roaming`, `block_sim`, `check_offers`, etc.) only
  have ~9-25 training examples each — thin by ML standards.
- **Real English examples pulled from the 150-intent dataset**, but *only* for
  intents whose phrasing genuinely generalizes to telecom IVR use (greetings,
  yes/no, cancel, repeat, fraud reporting, calling, "are you a bot") — relabeled
  where needed (`balance`→`check_balance`, `make_call`→`call_contact`). No
  bank-specific phrasing (e.g. "direct deposit," "APR") was force-fit into telecom
  intents it doesn't actually belong to.

## Evaluation results (honest numbers, not cherry-picked)

**General model (150 intents, real data):**
- Test accuracy: **92.3%**, macro F1: 0.923, cross-val: 0.923 (±0.005) — stable.
- Weakest intents (F1 0.72–0.81): `maybe`, `cancel`, `goodbye`, `yes`, `greeting`,
  `income`, `calendar_update`, `make_call`. These are short, generic phrases that
  genuinely overlap in wording with each other — a sensible, explainable failure
  mode, not a bug.

**Telecom model (22 intents, blended data):**
- Test accuracy: **93%**, weighted F1: 0.93, cross-val: 0.885 (±0.008).
- **Caveat worth taking seriously:** several Nigeria-specific intents
  (`activate_roaming`, `block_sim`, `check_offers`, `speak_to_agent`,
  `unblock_sim`) had only 2 examples in the held-out test set, because the
  underlying synthetic data for those intents is small (9-10 examples total).
  Their reported precision/recall (ranging 50-100%) is based on a sample too
  small to trust as a real performance estimate — it will look great or look
  bad almost at random until more real data is collected for these intents.
  **Before relying on this model for these specific intents in production,
  collect more real examples.** The well-represented intents (call_contact,
  check_balance, greeting, thank_you, cancel, repeat, report_fraud — all with
  100+ examples) have trustworthy metrics.

## Supported intents

**Telecom (22):** check_balance, recharge_airtime, check_data_balance, buy_data_bundle,
report_network_issue, speak_to_agent, block_sim, unblock_sim, change_plan, check_offers,
activate_roaming, report_fraud, general_complaint, call_contact, greeting, goodbye,
thank_you, yes, no, cancel, repeat, are_you_a_bot

**General (150):** run `GET /intents?model=general` or see `data/full_150_intents_en.csv`
for the full list — spans banking, travel, smart home, food, and general assistant chit-chat.

## Project structure

```
voice_command_classifier/
├── data/
│   ├── generate_data.py            # synthetic Pidgin/telecom data generator
│   ├── intents_dataset.csv         # generated synthetic telecom data (880 examples)
│   ├── full_150_intents_en.csv     # real dataset, English-only, French dropped
│   ├── build_telecom_dataset.py    # combines synthetic + real (overlapping intents)
│   └── telecom_dataset.csv         # final blended telecom training set (1315 examples)
├── models/
│   ├── train_telecom.py            # trains the telecom model
│   ├── train_full_150.py           # trains the general model
│   ├── telecom_pipeline.joblib     # trained telecom model
│   └── full_150_pipeline.joblib    # trained general model
├── app/
│   ├── model.py                     # inference wrapper, supports both models
│   └── stt.py                       # Whisper speech-to-text wrapper
├── api/
│   └── main.py                       # FastAPI backend, model param on each endpoint
├── streamlit_app.py                  # frontend with model selector
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

## Running locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
Whisper needs `ffmpeg` on your system (`brew install ffmpeg` / `apt install ffmpeg`).

### 2. (Optional) Regenerate data and retrain
Both trained models are already included, but to reproduce from scratch:
```bash
python data/generate_data.py           # regenerate synthetic telecom data
python data/build_telecom_dataset.py   # rebuild the blended telecom dataset
python models/train_telecom.py         # retrain telecom model
python models/train_full_150.py        # retrain general model
```

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for interactive API docs. Every request takes an
optional `model` field/param (`"telecom"` or `"general"`, defaults to `"telecom"`).

### 4. Start the UI
```bash
streamlit run streamlit_app.py
```
Visit `http://localhost:8501` — use the radio button at the top to switch models.

## Running with Docker
```bash
docker-compose up --build
```

## Deploying
See the deployment walkthrough discussed separately (Render for the API, Streamlit
Community Cloud for the UI). `render.yaml` is included for Render's Blueprint deploy.
Note `WHISPER_MODEL` env var defaults to `tiny` to fit free-tier RAM limits.

## Known limitations / honest caveats

- **Telecom model's Nigeria-specific intents are data-thin** (see Evaluation section
  above) — the headline accuracy number is real, but a handful of intents need more
  real-world data before the metrics on them can be trusted.
- **No entity extraction** — `call_contact` recognizes "call someone" but doesn't
  extract *who*; `recharge_airtime` doesn't extract the amount.
- **General model's weak intents are short/generic phrases** (yes/no/greeting/goodbye)
  that inherently overlap in wording — a modeling ceiling, not obviously fixable by
  more data alone.
- **Whisper accuracy on Nigerian-accented English/Pidgin hasn't been benchmarked** —
  worth testing manually before relying on it for Pidgin specifically.
- **Single-turn only** — no conversation memory or multi-turn clarification.

## Possible next steps

- Collect real (not synthetic) examples for the thin telecom intents — even 50-100
  real utterances per intent would meaningfully firm up those metrics.
- Fine-tune a transformer (DistilBERT/AfriBERTa) if GPU access is available — likely
  more robust to phrasing variation than TF-IDF, especially for the general model's
  150-way classification.
- Add entity/slot extraction for amount, plan, and contact-name intents.
- Benchmark Whisper specifically on Pidgin audio and consider a Pidgin-tuned STT
  alternative if accuracy is poor.
