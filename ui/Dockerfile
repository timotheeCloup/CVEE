FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

ENTRYPOINT ["sh", "-c"]
CMD ["streamlit run app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
