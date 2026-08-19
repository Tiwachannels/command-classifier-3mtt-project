"""
FastAPI backend for the Voice/Text Command Classifier.

Supports two models, selected via a `model` field/param ("telecom" or
"general"). Defaults to "telecom" - the Nigerian Pidgin-aware IVR classifier.

Endpoints:
  POST /predict/text   - classify a raw text command
  POST /predict/audio  - transcribe an audio file then classify it
  GET  /intents         - list supported intents for a given model
  GET  /health          - health check
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal

from app.model import predict_intent, list_intents

app = FastAPI(
    title="Voice/Text Command Classifier",
    description=(
        "Classifies commands from text or voice input. Two models available: "
        "'telecom' (Nigerian Pidgin-aware telecom IVR, 22 intents) and "
        "'general' (150-intent general-purpose assistant classifier)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ModelName = Literal["telecom", "general"]


class TextRequest(BaseModel):
    text: str
    model: ModelName = "telecom"


class PredictionResponse(BaseModel):
    intent: str
    confidence: float
    all_scores: dict
    transcript: str | None = None
    model: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/intents")
def get_intents(model: ModelName = Query("telecom")):
    return {"model": model, "intents": list_intents(model)}


@app.post("/predict/text", response_model=PredictionResponse)
def predict_text(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text field cannot be empty")
    result = predict_intent(payload.text, model_name=payload.model)
    return PredictionResponse(**result, transcript=payload.text, model=payload.model)


@app.post("/predict/audio", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...), model: ModelName = Query("telecom")):
    # Imported here so text-only usage never requires whisper/ffmpeg to be
    # installed or the model weights to be downloaded.
    from app.stt import transcribe_audio_bytes

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio file")

    try:
        stt_result = transcribe_audio_bytes(audio_bytes, filename_hint=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcription failed: {e}")

    transcript = stt_result["text"]
    if not transcript:
        raise HTTPException(status_code=422, detail="could not transcribe any speech from audio")

    result = predict_intent(transcript, model_name=model)
    return PredictionResponse(**result, transcript=transcript, model=model)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
