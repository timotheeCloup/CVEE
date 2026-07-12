# ── Stage 1: export the embedding model to ONNX (heavy deps, build-only) ──
FROM python:3.12-slim AS exporter

WORKDIR /export

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/export-venv
ENV PATH="/opt/export-venv/bin:$PATH"
ENV HF_HOME=/tmp/hf_cache

RUN uv venv /opt/export-venv
RUN uv pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --no-cache-dir "optimum[onnxruntime]"

COPY export_onnx.py .
RUN python export_onnx.py /export/onnx_model

# ── Stage 2: build the lean runtime virtualenv (no torch) ──
FROM python:3.12-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN uv venv /opt/venv
RUN uv pip install --no-cache-dir -r requirements.txt

# ── Stage 3: runtime ──
FROM python:3.12-slim AS runtime

RUN groupadd -r app && useradd -r -g app app

COPY --from=builder /opt/venv /opt/venv
COPY --from=exporter /export/onnx_model /app/onnx_model

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV ONNX_MODEL_DIR=/app/onnx_model
ENV PORT=8080

WORKDIR /app
COPY app.py embed_cv_search.py utils.py models.py config.py stopwords.json ./

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
