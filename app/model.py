"""
Loads a trained intent classifier and exposes predict_intent() with a
confidence threshold + fallback intent for low-confidence predictions.
Supports two models:
  - "telecom": Nigerian Pidgin-aware telecom IVR classifier (22 intents)
  - "general": full 150-intent general-purpose classifier
"""
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATHS = {
    "telecom": os.path.join(MODEL_DIR, "telecom_pipeline.joblib"),
    "general": os.path.join(MODEL_DIR, "full_150_pipeline.joblib"),
}
CONFIDENCE_THRESHOLD = 0.35
FALLBACK_INTENT = "unrecognized"

_pipelines = {}


def _load_pipeline(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from {list(MODEL_PATHS)}")
    if model_name not in _pipelines:
        _pipelines[model_name] = joblib.load(MODEL_PATHS[model_name])
    return _pipelines[model_name]


def predict_intent(text: str, model_name: str = "telecom") -> dict:
    """
    Returns a dict: {intent, confidence, all_scores}
    Falls back to 'unrecognized' if confidence is below threshold, rather
    than forcing a possibly-wrong guess.
    """
    if not text or not text.strip():
        return {"intent": FALLBACK_INTENT, "confidence": 0.0, "all_scores": {}}

    pipeline = _load_pipeline(model_name)
    probs = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_

    scored = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    top_intent, top_conf = scored[0]

    if top_conf < CONFIDENCE_THRESHOLD:
        intent = FALLBACK_INTENT
    else:
        intent = top_intent

    return {
        "intent": intent,
        "confidence": round(float(top_conf), 3),
        "all_scores": {k: round(float(v), 3) for k, v in scored[:5]},
    }


def list_intents(model_name: str = "telecom") -> list:
    pipeline = _load_pipeline(model_name)
    return sorted(pipeline.classes_.tolist())
