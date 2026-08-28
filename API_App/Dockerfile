# syntax=docker/dockerfile:1

# Python 3.12 matches the working local environment (.venv312) and the pinned
# wheels in requirements.txt.
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime required by XGBoost and ONNX Runtime.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/artifacts \
    PNEUMONIA_MODEL_DIR=/app/artifacts/advanced_pneumonia \
    EVIDENCES_JSON_PATH=/app/metadata/release_evidences.json \
    CONDITIONS_JSON_PATH=/app/metadata/release_conditions.json \
    EVIDENCE_DISPLAY_EN_PATH=/app/metadata/evidence_display_en.json

# Dependencies first: editing the application does not rebuild this layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Unprivileged runtime user. The application only reads its artifacts.
RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# Standard-library probe, so the image does not need curl or wget.
# The long start period covers loading the sklearn/XGBoost and ONNX artifacts.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# One worker on purpose: every additional worker would load the ML models again.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
