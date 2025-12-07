from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
from utils import extract_text_from_pdf, search_jobs_vector
from embed_cv_search import embed_cv_and_search
import os
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")


app = FastAPI(title="CV-Embedding Engine API")

@app.post("/embed-cv")
async def embed_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be PDF")
    
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    
    # Appel service embedding
    response = requests.post(EMBEDDING_API_URL, json={"text": text})
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Embedding service error")
    
    embedding = response.json()["embedding"]
    top_jobs = embed_cv_and_search(embedding)
    return {"top_jobs": top_jobs}

