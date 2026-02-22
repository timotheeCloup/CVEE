import os
import io
import PyPDF2
import psycopg2
from dotenv import load_dotenv
import math
import re
import time

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

TOP_K = 100

# Hybrid search weights (easily configurable)
EMBEDDING_WEIGHT = 1
FTS_WEIGHT = 2

# FTS weights for tsvector levels (A=Title, B=Description, C=Skills/Qualities)
# Higher values = higher priority
FTS_WEIGHTS = [0.3, 0.6, 1.0]  # [C, B, A]


def extract_text_from_pdf(file_bytes):
    """Extract text from first page of PDF"""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    if reader.pages:  
        text = reader.pages[0].extract_text() or "" 
    return text.strip()


def extract_french_keywords_from_headline(headline):
    """
    Extract French keywords from ts_headline fragment.
    PostgreSQL ts_headline uses <b>...</b> tags for FTS matches.
    
    Returns:
        List of clean keywords from the headline
    """
    if not headline:
        return []
    
    # Extract terms marked with <b>...</b> tags
    marked_terms = re.findall(r'<b>([^<]+)</b>', str(headline))
    
    keywords = []
    seen = set()
    
    for term in marked_terms:
        clean_term = term.strip().lower()
        # Remove punctuation but keep spaces for multi-word terms
        clean_term = re.sub(r'[^\w\s]', '', clean_term)
        
        # Filter very short words
        words = [w for w in clean_term.split() if len(w) > 2]
        if words:
            final_term = ' '.join(words)
            if final_term not in seen:
                keywords.append(final_term)
                seen.add(final_term)
    
    return keywords[:10]  # Top 10 keywords


def linear_mapping(value, from_min, from_max, to_min, to_max):
    """Linearly map a value from one range to another"""
    if from_max - from_min == 0:
        return to_min  # Avoid division by zero
    return to_min + (to_max - to_min) * (value - from_min) / (from_max - from_min)

def search_jobs_vector_hybrid(embedding, cv_text_fts, cv_text_orig):
    """
    Hybrid job search combining FTS (French) and semantic embedding (English).
    
    Process:
    1. Full-Text Search on French job descriptions
       - Uses ts_rank with custom weights for tsvector levels (A, B, C)
       - Level A (title) = highest priority = 1.0
       - Level B (description) = high priority = 0.6
       - Level C (skills/qualities) = medium priority = 0.3
       - Extracts top 10 keywords from each result
    2. Semantic search using embedding
       - Calculates cosine distance
    3. Combines scores: EMBEDDING_WEIGHT * embedding_score + FTS_WEIGHT * fts_score
    4. Returns top 100 results sorted by combined score
    
    Args:
        embedding: Vector embedding of translated CV (English)
        cv_text_fts: CV text with French stopwords removed (for FTS query)
        cv_text_orig: Original CV text (French)
    
    Returns:
        List of top 100 matched jobs with scores
    
    Configuration (top of utils.py):
        EMBEDDING_WEIGHT: Weight for semantic similarity (0.0-1.0)
        FTS_WEIGHT: Weight for full-text search (0.0-1.0)
        FTS_WEIGHTS: Custom weights for tsvector levels [C, B, A]
    """
    t_db_start = time.time()
    
    # Connect to database
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cur = conn.cursor()
    t_conn = time.time()
    print(f"   - DB connection: {t_conn - t_db_start:.3f}s", flush=True)
    
    print(f"   - CV text for FTS: {len(cv_text_fts)} chars", flush=True)
    print(f"   - Embedding dimension: {len(embedding)}", flush=True)
    
    # Build French FTS query from CV text
    # Convert space-separated terms into PostgreSQL tsquery format
    fts_terms = cv_text_fts.split()[:1000]  # Limit to 50 terms for safety
    if fts_terms:
        # Use OR operator to match ANY of the CV terms
        tsquery = " | ".join([f"'{term}'" for term in fts_terms])
    else:
        tsquery = "'offre'"  # Fallback
    
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    
    # Retrieve results with both embedding similarity and FTS ranking
    # Calculate combined score: EMBEDDING_WEIGHT * semantic + FTS_WEIGHT * fts
    # ts_rank expects 4 weights for levels [D, C, B, A] (D=0 since unused)
    sql = """
    SELECT 
        jg.job_id,
        (1 - (jg.embedding <-> %s))::float8 as embedding_score,
        COALESCE(ts_rank('{0, 0.3, 0.6, 1.0}'::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8 as fts_score,
        (%s * (1 - (jg.embedding <-> %s))::float8 + 
         %s * COALESCE(ts_rank('{0, 0.3, 0.6, 1.0}'::float4[], jg.fts_tokens, to_tsquery('french', %s)), 0)::float8)::float8 as combined_score,
        js.intitule,
        js.entreprise->>'nom' AS entreprise,
        js.lieuTravail->>'libelle' AS lieu,
        js.typeContratLibelle,
        js.dateCreation,
        ts_headline('french', 
            js.intitule || ' ' || COALESCE(js.description, '') || ' ' ||
            COALESCE((SELECT string_agg(elem->>'libelle', ' ') 
                      FROM jsonb_array_elements(js.competences) AS elem), '') || ' ' ||
            COALESCE((SELECT string_agg((elem->>'libelle') || ' ' || (elem->>'description'), ' ') 
                      FROM jsonb_array_elements(js.qualitesprofessionnelles) AS elem), ''),
            to_tsquery('french', %s), 
            'StartSel=<b>, StopSel=</b>, MaxWords=100, MinWords=50') as headline
    FROM jobs_gold jg
    JOIN jobs_silver js ON jg.job_id = js.job_id
    WHERE jg.fts_tokens IS NOT NULL
    ORDER BY combined_score DESC
    LIMIT %s;
    """
    
    cur.execute(sql, (
        embedding_str, tsquery, 
        EMBEDDING_WEIGHT, embedding_str, 
        FTS_WEIGHT, tsquery, 
        tsquery, 
        TOP_K
    ))
    results = cur.fetchall()
    t_query = time.time()
    print(f"   - Query execution: {t_query - t_conn:.3f}s (retrieved {len(results)} jobs)", flush=True)
    
    cur.close()
    conn.close()
    
    # Process results (already sorted by combined score from database)
    hybrid_results = []
    
    for r in results:
        (job_id, embedding_score, fts_score, combined_score, intitule, entreprise, lieu, 
         type_contrat, date_creation, headline) = r
        
        # Extract keywords from headline
        keywords = extract_french_keywords_from_headline(headline)
        
        hybrid_results.append({
            'job_id': job_id,
            'embedding_score': embedding_score,
            'fts_score': fts_score,
            'combined_score': combined_score,
            'intitule': intitule,
            'entreprise': entreprise,
            'lieu': lieu,
            'type_contrat': type_contrat,
            'date_creation': date_creation,
            'keywords': keywords
        })
    
    t_process = time.time()
    print(f"   - Processing: {t_process - t_query:.3f}s", flush=True)
    
    # Log results summary
    fts_scores = [r['fts_score'] for r in hybrid_results]
    fts_non_zero = sum(1 for score in fts_scores if score > 0)
    print(f"\n   === RESULTS SUMMARY ===", flush=True)
    print(f"   - Hybrid weights: {EMBEDDING_WEIGHT * 100:.0f}% embedding + {FTS_WEIGHT * 100:.0f}% FTS", flush=True)
    print(f"   - FTS level weights (C,B,A): {FTS_WEIGHTS}", flush=True)
    print(f"   - Results with FTS score > 0: {fts_non_zero}/{len(hybrid_results)}", flush=True)
    
    # Log top 5 results by combined score
    print(f"\n   === TOP 5 RESULTS (by combined score) ===", flush=True)
    for i in range(min(5, len(hybrid_results))):
        r = hybrid_results[i]
        print(f"   #{i+1:2d} | Emb: {r['embedding_score']:.4f} | FTS: {r['fts_score']:.4f} | Combined: {r['combined_score']:.4f} | {r['intitule'][:20]} | {r['job_id']}", flush=True)
    
    # Log bottom 5 results by combined score
    print(f"\n   === BOTTOM 5 RESULTS (by combined score) ===", flush=True)
    for i in range(len(hybrid_results) - 1, max(0, len(hybrid_results) - 5) - 1, -1):
        r = hybrid_results[i]
        print(f"   #{i+1:2d} | Emb: {r['embedding_score']:.4f} | FTS: {r['fts_score']:.4f} | Combined: {r['combined_score']:.4f} | {r['intitule'][:20]} | {r['job_id']}", flush=True)
    
    # Log top 5 embedding scores
    print(f"\n   === TOP 5 BY EMBEDDING SCORE ===", flush=True)
    sorted_by_emb = sorted(hybrid_results, key=lambda x: x['embedding_score'], reverse=True)
    for i in range(min(5, len(sorted_by_emb))):
        r = sorted_by_emb[i]
        print(f"   #{i+1:2d} | Emb: {r['embedding_score']:.4f} | FTS: {r['fts_score']:.4f} | Combined: {r['combined_score']:.4f} | {r['intitule'][:20]} | {r['job_id']}", flush=True)
    
    # Log bottom 5 embedding scores
    print(f"\n   === BOTTOM 5 BY EMBEDDING SCORE ===", flush=True)
    for i in range(min(5, len(sorted_by_emb))):
        r = sorted_by_emb[-(i+1)]
        print(f"   #{i+1:2d} | Emb: {r['embedding_score']:.4f} | FTS: {r['fts_score']:.4f} | Combined: {r['combined_score']:.4f} | {r['intitule'][:20]} | {r['job_id']}", flush=True)
    
    # Log top 5 FTS scores
    print(f"\n   === TOP 5 BY FTS SCORE ===", flush=True)
    sorted_by_fts = sorted(hybrid_results, key=lambda x: x['fts_score'], reverse=True)
    for i in range(min(5, len(sorted_by_fts))):
        r = sorted_by_fts[i]
        print(f"   #{i+1:2d} | Emb: {r['embedding_score']:.4f} | FTS: {r['fts_score']:.4f} | Combined: {r['combined_score']:.4f} | {r['intitule'][:20]} | {r['job_id']}", flush=True)
    
    # Log bottom 5 FTS scores
    print(f"\n   === BOTTOM 5 BY FTS SCORE ===", flush=True)
    for i in range(min(5, len(sorted_by_fts))):
        r = sorted_by_fts[-(i+1)]
        print(f"   #{i+1:2d} | Emb: {r['embedding_score']:.4f} | FTS: {r['fts_score']:.4f} | Combined: {r['combined_score']:.4f} | {r['intitule'][:20]} | {r['job_id']}", flush=True)
    
    # Format results for API response
    processed_results = []
    for r in hybrid_results:
        processed_results.append({
            "job_id": r['job_id'],
            "similarity_score": round(linear_mapping(r['combined_score'], 0.45, 0.7, 0.2, 0.8), 2),
            "embedding_score": round(r['embedding_score'], 4),
            "fts_score": round(r['fts_score'], 4),
            "combined_score": round(r['combined_score'], 4),
            "intitule": r['intitule'],
            "entreprise": r['entreprise'],
            "lieu": r['lieu'],
            "type_contrat": r['type_contrat'],
            "date_creation": r['date_creation'],
            "matching_terms": r['keywords']
        })
    
    return processed_results


