"""
Speech-to-text wrapper using OpenAI Whisper.

NOTE: Whisper downloads its model weights on first use from OpenAI's CDN.
This requires normal outbound internet access on whatever machine runs it
(your laptop, or the deployed server) — it does not require an API key,
Whisper runs fully locally once the weights are cached.

We use the 'base' model as a speed/accuracy tradeoff for short commands.
Swap to 'tiny' for faster/lower-accuracy, or 'small'/'medium' for better
accuracy at the cost of speed and memory.
"""
import whisper
import tempfile
import os

_model = None
# Override with the WHISPER_MODEL env var if needed. 'tiny' uses far less
# RAM (~1GB total vs ~2GB+ for 'base') which matters on free-tier hosting
# like Render's 512MB-1GB instances. Accuracy trade-off: tiny is noticeably
# less accurate, especially on accented speech - fine for a demo, worth
# upgrading to 'base' or 'small' once you're on a paid tier.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")


def _load_model():
    global _model
    if _model is None:
        _model = whisper.load_model(MODEL_SIZE)
    return _model


def transcribe_audio_bytes(audio_bytes: bytes, filename_hint: str = "audio.wav") -> dict:
    """
    Accepts raw audio bytes (wav/mp3/m4a/etc — whatever ffmpeg can decode),
    writes to a temp file, and returns Whisper's transcription result.
    """
    model = _load_model()
    suffix = os.path.splitext(filename_hint)[1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, fp16=False)
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
        }
    finally:
        os.remove(tmp_path)
