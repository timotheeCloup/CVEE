FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .
RUN uv pip install --system --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENTRYPOINT ["sh", "-c"]
CMD ["streamlit run app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
