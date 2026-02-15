from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import time
from utils import extract_text_from_pdf
from embed_cv_search import embed_cv_and_search

app = FastAPI(title="CV-Embedding Engine API")

@app.get("/health")
async def health():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy"}

@app.post("/embed-cv")
async def embed_cv(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be PDF")
    
    t_start = time.time()
    
    file_bytes = await file.read()
    t_read = time.time()
    print(f"File read: {t_read - t_start:.2f}s", flush=True)
    
    text = extract_text_from_pdf(file_bytes)
    t_extract = time.time()
    print(f"PDF extract: {t_extract - t_read:.2f}s", flush=True)
    
    top_jobs = embed_cv_and_search(text)
    t_end = time.time()
    print(f"Total time: {t_end - t_start:.2f}s\n", flush=True)
    
    return {"top_jobs": top_jobs}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)