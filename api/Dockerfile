FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8080

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/app/models_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en')"

COPY . .

EXPOSE 8080
ENTRYPOINT ["sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]