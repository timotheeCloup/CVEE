import os
import io
import PyPDF2
import psycopg2
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

TOP_K = 100

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:  
        text = reader.pages[0].extract_text() or "" 
    return text.strip()

def get_all_matching_terms_batch(cv_text, job_descriptions, top_n=10):
    """
    Extract matching terms between CV and MULTIPLE job offers using TF-IDF.
    Much faster than doing it one by one (vectorizes CV once instead of 100 times).
    
    Args:
        cv_text: The CV text
        job_descriptions: List of job description strings
        top_n: Number of top terms to extract per job
    
    Returns:
        List of lists of matching terms (one list per job)
    """
    cv_text = str(cv_text).lower()
    job_descriptions = [str(j).lower() for j in job_descriptions]
    
    # ALL texts: CV + jobs
    all_texts = [cv_text] + job_descriptions
    
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        cv_vec = tfidf_matrix[0].toarray()[0]
        
        results = []
        # For each job (indices 1 to len(job_descriptions))
        for job_idx in range(1, len(all_texts)):
            job_vec = tfidf_matrix[job_idx].toarray()[0]
            
            # Find matching terms
            matching_scores = []
            for i, feature in enumerate(feature_names):
                cv_score = cv_vec[i]
                job_score = job_vec[i]
                if cv_score > 0 and job_score > 0:
                    matching_scores.append({
                        "term": feature,
                        "score": cv_score * job_score
                    })
            
            # Sort and get top N
            matching_scores.sort(key=lambda x: x["score"], reverse=True)
            top_terms = [item["term"] for item in matching_scores[:top_n]]
            results.append(top_terms)
        
        return results
    except Exception as e:
        print(f"Error in get_all_matching_terms_batch: {str(e)}")
        return [[] for _ in job_descriptions]

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
        js.vector_text_input
    FROM jobs_gold jg
    JOIN jobs_silver js ON jg.job_id = js.job_id
    ORDER BY jg.embedding <-> %s
    LIMIT %s;
    """
    cur.execute(sql, (embedding_str, embedding_str, top_k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    
    # Extract all job descriptions first
    job_descriptions = [r[7] if r[7] else "" for r in results]
    
    # Calculate all matching terms in ONE BATCH (much faster!)
    all_matching_terms = get_all_matching_terms_batch(cv_text, job_descriptions, top_n=10)
    
    # Process results
    processed_results = []
    for i, r in enumerate(results):
        processed_results.append({
            "job_id": r[0],
            "similarity_score": round(r[1], 4),
            "intitule": r[2],
            "entreprise": r[3],
            "lieu": r[4],
            "type_contrat": r[5],
            "date_creation": r[6],
            "matching_terms": all_matching_terms[i]
        })
    
    return processed_results