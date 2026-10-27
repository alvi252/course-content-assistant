# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LIBRARY_DIR=/app/data/library \
    VOICE_DIR=/app/data/voice \
    INDEX_PATH=/app/var/index.sqlite

WORKDIR /app

# Install the package with its dependencies (and the optional PDF loader).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[pdf]"

COPY data/library ./data/library
COPY data/voice ./data/voice

RUN useradd --create-home --uid 10001 app && mkdir -p /app/var && chown app /app/var
USER app

# The index and embedding cache live in /app/var: keep it on a volume so restarts don't
# re-embed anything. Mount your own lessons over /app/data/library (and your style sheet and sample posts over
# /app/data/voice) to serve them instead.
VOLUME ["/app/var"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# Ingest on start (only new or changed documents are embedded), then serve.
CMD ["sh", "-c", "course-assistant ingest && course-assistant serve --host 0.0.0.0 --port 8000"]
