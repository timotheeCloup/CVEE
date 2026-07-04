FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN uv venv /opt/venv
RUN uv pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir -r requirements.txt

ENV HF_HOME=/app/models_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('antoinelouis/french-me5-small')"

FROM python:3.12-slim AS runtime

RUN groupadd -r app && useradd -r -g app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models_cache /app/models_cache

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models_cache
ENV PORT=8080

WORKDIR /app
COPY app.py embed_cv_search.py utils.py stopwords.json ./

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
