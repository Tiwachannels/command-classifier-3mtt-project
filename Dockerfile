FROM python:3.11-slim

# ffmpeg is required by Whisper to decode audio formats (mp3, m4a, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the Whisper model at build time so the first request isn't
# slow. Defaults to 'tiny' to fit comfortably on free-tier hosts (Render
# free tier = 512MB RAM). Override with --build-arg WHISPER_MODEL=base for
# better accuracy once you're on a paid tier with more RAM.
ARG WHISPER_MODEL=tiny
ENV WHISPER_MODEL=${WHISPER_MODEL}
RUN python -c "import os, whisper; whisper.load_model(os.environ['WHISPER_MODEL'])"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
