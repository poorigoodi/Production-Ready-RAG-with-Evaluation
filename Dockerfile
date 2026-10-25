FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY scripts ./scripts
COPY data/evaluation ./data/evaluation
COPY data/documents/sample ./data/documents/sample
COPY README.md LICENSE ./

RUN mkdir -p /app/data/documents /app/data/indexes /app/reports/metrics /app/reports/figures /models

EXPOSE 8000 8501

# Default command is overridden in docker-compose.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
