# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/build/models_cache

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Pre-download the model (takes time but cached after first build)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en')"

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONOPTIMIZE=1
ENV PORT=8080
ENV HF_HOME=/app/models_cache

# Copy only necessary files from builder
# pip install --user puts packages in /root/.local regardless of WORKDIR
COPY --from=builder /root/.local /root/.local
COPY --from=builder /build/models_cache /app/models_cache

# Setup PATH
ENV PATH=/root/.local/bin:$PATH

# Application code
COPY app.py embed_cv_search.py utils.py stopwords.json ./

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]