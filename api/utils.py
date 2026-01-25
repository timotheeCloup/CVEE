import os
import io
import re
import PyPDF2
import psycopg2
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

TOP_K = 21

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:  
        text = reader.pages[0].extract_text() or "" 
    return text.strip()

def get_top_matching_terms(cv_text, job_text, top_n=10):
    """
    Extract top N terms that matched between CV and job offer using TF-IDF similarity
    """
    # Clean and tokenize
    cv_text = cv_text.lower()
    job_text = job_text.lower()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=1
    )
    
    try:
        # Fit on both texts
        tfidf_matrix = vectorizer.fit_transform([cv_text, job_text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the vectors
        cv_vec = tfidf_matrix[0].toarray()[0]
        job_vec = tfidf_matrix[1].toarray()[0]
        
        # Find matching terms (where both have non-zero values)
        matching_scores = []
        for i, feature in enumerate(feature_names):
            cv_score = cv_vec[i]
            job_score = job_vec[i]
            # Both terms present in CV and job
            if cv_score > 0 and job_score > 0:
                matching_scores.append({
                    "term": feature,
                    "score": cv_score * job_score
                })
        
        # Sort and get top N
        matching_scores.sort(key=lambda x: x["score"], reverse=True)
        top_terms = [item["term"] for item in matching_scores[:top_n]]
        
        return top_terms
    except:
        return []

def search_jobs_vector(embedding, cv_text="", top_k=TOP_K):
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
        js.dateCreation,
        js.description
    FROM jobs_gold jg
    JOIN jobs_silver js ON jg.job_id = js.job_id
    ORDER BY jg.embedding <-> %s
    LIMIT %s;
    """
    cur.execute(sql, (embedding_str, embedding_str, top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Process results and add matching terms
    processed_results = []
    for r in results:
        job_description = r[7] if r[7] else ""
        matching_terms = get_top_matching_terms(cv_text, job_description, top_n=10)
        
        processed_results.append({
            "job_id": r[0],
            "similarity_score": round(r[1], 4),
            "intitule": r[2],
            "entreprise": r[3],
            "lieu": r[4],
            "type_contrat": r[5],
            "date_creation": r[6],
            "matching_terms": matching_terms
        })
    
    return processed_results