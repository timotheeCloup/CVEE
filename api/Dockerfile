FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/models_cache
ENV PORT=8080

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN uv pip install --system --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('antoinelouis/french-me5-small')"

COPY app.py embed_cv_search.py utils.py stopwords.json ./

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
