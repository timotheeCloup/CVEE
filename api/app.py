from fastapi import FastAPI, UploadFile, File, HTTPException
from utils import extract_text_from_pdf
from embed_cv_search import embed_cv_and_search



app = FastAPI(title="CV-Embedding Engine API")

@app.post("/embed-cv")
async def embed_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be PDF")
    
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    
    top_jobs = embed_cv_and_search(text)
    return {"top_jobs": top_jobs}

