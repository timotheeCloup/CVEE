FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HF_HOME=/app/models_cache

# LAYER 1: Dependencies (rarely changes) - install and cache model
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the model (takes time but cached after first build)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en')"

# LAYER 2: Application code (changes frequently)
COPY app.py embed_cv_search.py utils.py stopwords.json ./

EXPOSE 8080
ENTRYPOINT ["sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]