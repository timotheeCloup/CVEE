import os
import io
import PyPDF2
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

TOP_K = 20

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:  
        text = reader.pages[0].extract_text() or "" 
    return text.strip()

def search_jobs_vector(embedding, top_k=TOP_K):
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cur = conn.cursor()
    
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    
    sql = f"""
    SELECT 
        jg.job_id,
        1 - (jg.embedding <-> %s) AS similarity_score,
        js.intitule,
        js.entreprise->>'nom' AS entreprise,
        js.lieuTravail->>'libelle' AS lieu,
        js.typeContratLibelle,
        js.dateCreation
    FROM jobs_gold jg
    JOIN jobs_silver js ON jg.job_id = js.job_id
    ORDER BY jg.embedding <-> %s
    LIMIT %s;
    """
    cur.execute(sql, (embedding_str, embedding_str, top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "job_id": r[0],
            "similarity_score": round(r[1], 4),
            "intitule": r[2],
            "entreprise": r[3],
            "lieu": r[4],
            "type_contrat": r[5],
            "date_creation": r[6]
        } for r in results
    ]