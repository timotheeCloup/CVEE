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


def _generate_ngrams(tokens, n):
    """
    Generate n-grams from a list of tokens.
    
    Args:
        tokens: List of tokens/words
        n: Size of n-gram (e.g., 3 for trigrams)
    
    Returns:
        List of n-grams as tuples
    """
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def extract_cv_terms_tfidf(cv_text, top_n=None):
    """
    Extract ALL significant terms from CV text (no arbitrary limit).
    Works with short single-document texts.
    
    Args:
        cv_text: Full CV text
        top_n: (deprecated) kept for backward compatibility, ignored
    
    Returns:
        Set of ALL significant CV terms (filtered by stopwords and length only)
    """
    if not cv_text or len(cv_text.strip()) < 10:
        return set()
    
    try:
        # Simple word frequency approach for single document
        from collections import Counter
        import re
        
        # Common English stopwords
        stopwords = {
            'the', 'and', 'or', 'is', 'to', 'of', 'in', 'at', 'for', 'on', 'with', 'by', 'from',
            'are', 'be', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'must', 'as', 'a', 'an', 'this', 'that',
            'these', 'those', 'it', 'its', 'he', 'she', 'they', 'them', 'an', 'your', 'his', 'her'
        }
        
        # Simple tokenization
        words = re.findall(r'\b[a-z]+\b', cv_text.lower())
        # Filter stopwords and short words - RETURN ALL significant terms
        filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Return all unique significant terms (no limit)
        return set(filtered_words)
    except Exception as e:
        print(f"   - Term extraction error: {e}", flush=True)
        return set()


def extract_true_matching_terms_from_headline(headlines, cv_terms, top_n=10):
    """
    Extract REAL matching terms from ts_headline fragments using the <b>...</b> markers.
    PostgreSQL ts_headline() marks the actual full-text search matches with <b>...</b> tags.
    These are the ONLY terms we extract - no guessing!
    Generates n-grams (1-3) from the extracted terms.
    
    Args:
        headlines: List of headline strings from ts_headline() with <b>match</b> markers
        cv_terms: Set of CV terms (used only for filtering, not matching)
        top_n: Number of top matching terms/n-grams to return per job
    
    Returns:
        List of lists of matching terms with n-grams 1-3 (top N for each job)
    """
    import re
    
    results = []
    
    for headline in headlines:
        if not headline:
            results.append([])
            continue
        
        # Extract terms marked with <b>...</b> tags by ts_headline()
        # These are the ACTUAL full-text search matches from PostgreSQL
        marked_terms = re.findall(r'<b>([^<]+)</b>', str(headline))
        
        if not marked_terms:
            results.append([])
            continue
        
        # Clean and normalize the terms
        matching = []
        for term in marked_terms:
            # Remove extra whitespace and convert to lowercase
            clean_term = term.strip().lower()
            
            # Remove punctuation but keep spaces within multi-word terms
            clean_term = re.sub(r'[^\w\s]', '', clean_term)
            
            # Split into words if multi-word, filter very short words
            words = [w for w in clean_term.split() if len(w) > 2]
            
            if words:
                # Rejoin the cleaned words
                final_term = ' '.join(words)
                if final_term not in matching:
                    matching.append(final_term)
        
        # Generate n-grams (1-3) from the matched terms to create multi-word key phrases
        matching_with_ngrams = []
        seen = set()
        
        # Add individual terms first (1-grams)
        for term in matching:
            if term not in seen:
                matching_with_ngrams.append(term)
                seen.add(term)
        
        # Split matching terms into words for n-gram generation
        all_words = []
        for term in matching:
            all_words.extend(term.split())
        
        # Generate 2-grams and 3-grams from consecutive matched words
        for ngsize in range(2, 4):  # 2-grams and 3-grams
            ngrams = _generate_ngrams(all_words, ngsize)
            for ngram in ngrams:
                ngram_str = ' '.join(ngram)
                if ngram_str not in seen:
                    matching_with_ngrams.append(ngram_str)
                    seen.add(ngram_str)
        
        # Return top N unique matching terms (including n-grams)
        results.append(matching_with_ngrams[:top_n])
    
    return results


def extract_matching_terms_from_headline(headlines, cv_terms=None, top_n=20, max_ngram_size=3):
    """
    Extract matching terms from ts_headline fragments.
    FILTERED: Only returns terms that appear in both headline AND cv_terms (TF-IDF filtered).
    If cv_terms is empty/None, returns all terms without filtering.
    
    Args:
        headlines: List of headline strings from ts_headline()
        cv_terms: Set of significant CV terms from TF-IDF. If None or empty, no filtering applied.
        top_n: Number of top terms to extract per job (default: 20)
        max_ngram_size: Maximum n-gram size to extract (default: 3)
    
    Returns:
        List of lists of matching terms (filtered by CV relevance if cv_terms provided)
    """
    english_stopwords = {'the', 'and', 'or', 'is', 'to', 'of', 'in', 'at', 'for', 'on', 'with', 'by', 'from', 'are', 'be', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'as', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they', 'them'}
    
    results = []
    
    for headline in headlines:
        if not headline:
            results.append([])
            continue
        
        text = str(headline).lower()
        words = text.split()
        
        # Clean words: remove punctuation
        clean_words = []
        for word in words:
            clean_word = word.strip('.,;:!?()[]{}\"\'')
            if len(clean_word) > 2 and clean_word not in english_stopwords:
                clean_words.append(clean_word)
        
        if not clean_words:
            results.append([])
            continue
        
        terms = []
        seen = set()
        
        # Extract all n-grams from 1 to max_ngram_size (preserving order: 1-grams, 2-grams, 3-grams)
        for ngram_size in range(1, min(max_ngram_size + 1, len(clean_words) + 1)):
            ngram_list = _generate_ngrams(clean_words, ngram_size)
            
            for ngram in ngram_list:
                term = ' '.join(ngram)
                
                # FILTER: Only keep terms that are in CV TF-IDF terms (if cv_terms provided AND not empty)
                if cv_terms and term not in cv_terms:
                    continue
                
                if term not in seen:
                    terms.append(term)
                    seen.add(term)
        
        # Keep top N terms
        results.append(terms[:top_n])
    
    return results

def calculate_idf_batch(cv_terms, cur, total_docs):
    """
    Calculate IDF for ALL CV terms in a single batch query (instead of per-term queries).
    
    Args:
        cv_terms: Set of CV terms
        cur: Database cursor
        total_docs: Total number of documents in database
    
    Returns:
        Dict of term -> IDF score
    """
    import math
    
    if not cv_terms or total_docs == 0:
        return {}
    
    term_idf_cache = {}
    cv_terms_list = list(cv_terms)
    
    try:
        # Build a UNION query to get document counts for all terms in one shot
        query_parts = []
        params = []
        
        for term in cv_terms_list:
            query_parts.append(
                "SELECT %s as term, COUNT(DISTINCT job_id) as doc_count "
                "FROM jobs_gold WHERE fts_tokens @@ to_tsquery('english', %s)"
            )
            params.append(term)
            params.append(term)
        
        if query_parts:
            sql = " UNION ALL ".join(query_parts)
            try:
                cur.execute(sql, params)
                results = cur.fetchall()
                
                for term, doc_count in results:
                    if doc_count > 0:
                        idf = math.log(total_docs / max(doc_count, 1))
                        term_idf_cache[term] = idf
                    else:
                        term_idf_cache[term] = 1.0
                
                print(f"   - IDF calculated for {len(term_idf_cache)}/{len(cv_terms_list)} terms in single batch query", flush=True)
            except Exception as e:
                # Fallback: if batch query fails, use default values
                print(f"   - Batch IDF query failed: {e}, using default IDF", flush=True)
                for term in cv_terms_list:
                    term_idf_cache[term] = 1.0
    except Exception as e:
        print(f"   - IDF batch calculation error: {e}", flush=True)
        # Fallback: assign default IDF to all terms
        for term in cv_terms_list:
            term_idf_cache[term] = 1.0
    
    return term_idf_cache


def calculate_bm25_score(fts_tokens_str, cv_terms, term_idf_cache):
    """
    Calculate BM25 score by matching fts_tokens with CV terms.
    Now uses IDF for ALL CV terms (not limited to top 20).
    
    Args:
        fts_tokens_str: tsvector format string (e.g., "'data':3,15 'engineer':4,16")
        cv_terms: Set of terms extracted from CV
        term_idf_cache: Dict of term -> IDF scores
    
    Returns:
        float: BM25-like score based on matching terms rarity
    """
    if not fts_tokens_str or not cv_terms:
        return 0.0
    
    import re
    
    # Parse fts_tokens format: extract all terms (word before the colon)
    terms_in_fts = re.findall(r"'([^']+)'", fts_tokens_str)
    
    # Find matching terms and sum their IDF scores
    bm25_score = 0.0
    matched_count = 0
    
    for cv_term in cv_terms:
        # Direct match
        if cv_term in terms_in_fts:
            # Get IDF from cache or use default
            idf = term_idf_cache.get(cv_term, 1.0)
            bm25_score += idf
            matched_count += 1
        else:
            # Try partial match (first 3 chars)
            for fts_term in terms_in_fts:
                if len(cv_term) > 2 and fts_term.startswith(cv_term[:3]):
                    idf = term_idf_cache.get(fts_term, 0.5)  # Lower weight for partial match
                    bm25_score += idf * 0.5
                    break
    
    # Normalize by number of matches to avoid bias toward long CVs
    if matched_count > 0:
        bm25_score = bm25_score / matched_count
    
    return bm25_score

def search_jobs_vector(embedding, cv_text="", top_k=TOP_K):
    import time
    import re
    t_db_start = time.time()
    
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
    
    # Log CV text for debugging
    print(f"   - CV text length: {len(cv_text)} chars", flush=True)
    if cv_text:
        print(f"   - CV text preview: {cv_text[:100]}...", flush=True)
    
    # Extract CV terms - NOW RETURNS ALL TERMS (not limited to top 50)
    cv_terms = extract_cv_terms_tfidf(cv_text)
    print(f"   - CV terms: {len(cv_terms)} significant terms extracted", flush=True)
    if cv_terms:
        terms_list = sorted(list(cv_terms))[:20]  # Show first 20 alphabetically for debugging
        print(f"   - First 20 CV terms: {terms_list}", flush=True)
    
    # Calculate IDF for each CV term (rarity in DB) - NOW BATCH QUERY FOR ALL TERMS
    print(f"\n   === BM25 IDF CALCULATION ===", flush=True)
    total_docs = 0
    
    # Get total number of documents
    cur.execute("SELECT COUNT(*) FROM jobs_gold WHERE fts_tokens IS NOT NULL;")
    total_docs = cur.fetchone()[0]
    print(f"   - Total documents: {total_docs}", flush=True)
    
    # Calculate IDF for ALL CV terms in a single batch query (much faster)
    term_idf_cache = calculate_idf_batch(cv_terms, cur, total_docs)
    
    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
    
    # Build tsquery from ALL CV terms for full-text search with ts_headline
    # Create a query like: 'term1' & 'term2' | 'term3' etc.
    cv_terms_list = list(cv_terms)
    if cv_terms_list:
        # Use OR operator to find documents containing ANY of the CV terms
        tsquery = " | ".join([f"'{term}'" for term in cv_terms_list[:50]])  # Limit to 50 for SQL safety
    else:
        tsquery = "'job'"  # Fallback
    
    # Retrieve results with ts_headline for extracting real terms from original text
    sql = """
    SELECT 
        jg.job_id,
        (1 - (jg.embedding <-> %s))::float8 as semantic_score,
        js.intitule,
        js.entreprise->>'nom' AS entreprise,
        js.lieuTravail->>'libelle' AS lieu,
        js.typeContratLibelle,
        js.dateCreation,
        js.vector_text_input,
        jg.fts_tokens::text as fts_tokens,
        ts_headline('english', js.vector_text_input, to_tsquery('english', %s), 
                    'StartSel=<b>, StopSel=</b>, MaxWords=100, MinWords=50') as headline
    FROM jobs_gold jg
    JOIN jobs_silver js ON jg.job_id = js.job_id
    WHERE jg.fts_tokens IS NOT NULL
    ORDER BY (1 - (jg.embedding <-> %s)) DESC
    LIMIT %s;
    """
    
    cur.execute(sql, (embedding_str, tsquery, embedding_str, top_k))
    results = cur.fetchall()
    t_query = time.time()
    print(f"   - Query execution: {t_query - t_conn:.3f}s (retrieved {len(results)} jobs)", flush=True)
    
    cur.close()
    conn.close()
    
    # Calculate BM25 scores for each result
    results_with_bm25 = []
    for r in results:
        job_id, semantic_score, intitule, entreprise, lieu, type_contrat, date_creation, vector_text_input, fts_tokens, headline = r
        
        bm25_score = calculate_bm25_score(fts_tokens, cv_terms, term_idf_cache)
        
        # Calculate final score: 80% semantic + 20% BM25
        embedding_weight = 0.8
        keywords_weight = 0.2
        final_score = (semantic_score * embedding_weight) + (bm25_score * keywords_weight)
        
        results_with_bm25.append({
            'job_id': job_id,
            'semantic_score': semantic_score,
            'bm25_score': bm25_score,
            'final_score': final_score,
            'intitule': intitule,
            'entreprise': entreprise,
            'lieu': lieu,
            'type_contrat': type_contrat,
            'date_creation': date_creation,
            'vector_text_input': vector_text_input,
            'fts_tokens': fts_tokens,
            'headline': headline
        })
    
    # Re-sort by final_score (BM25 now matters)
    results_with_bm25.sort(key=lambda x: x['final_score'], reverse=True)
    t_calc = time.time()
    print(f"   - BM25 calculation: {t_calc - t_query:.3f}s", flush=True)
    
    # Extract all headlines for term extraction (real, non-stemmed terms)
    headlines_list = [r['headline'] if r['headline'] else "" for r in results_with_bm25]
    
    # Extract matching terms from ts_headline (REAL terms from original text with n-grams)
    all_matching_terms = extract_true_matching_terms_from_headline(
        headlines_list, 
        cv_terms=cv_terms,
        top_n=10
    )
    t_matching = time.time()
    print(f"   - Matching terms extraction: {t_matching - t_calc:.3f}s", flush=True)
    
    # Log results statistics
    bm25_scores = [r['bm25_score'] for r in results_with_bm25]
    bm25_non_zero = sum(1 for score in bm25_scores if score > 0)
    print(f"\n   === RESULTS SUMMARY ===", flush=True)
    print(f"   - Results with BM25 > 0: {bm25_non_zero}/{len(results_with_bm25)}", flush=True)
    
    # Log top 5 and bottom 5 results
    print(f"\n=== TOP 5 SEARCH RESULTS ===", flush=True)
    for i in range(min(5, len(results_with_bm25))):
        r = results_with_bm25[i]
        print(f"#{i+1:2d} | {r['job_id']:8s} | Semantic: {r['semantic_score']:.4f} | BM25: {r['bm25_score']:.4f} | Final: {r['final_score']:.4f} | {r['intitule'][:40]}", flush=True)
    print("", flush=True)
    
    print(f"=== BOTTOM 5 SEARCH RESULTS ===", flush=True)
    start_idx = max(0, len(results_with_bm25) - 5)
    for i in range(start_idx, len(results_with_bm25)):
        r = results_with_bm25[i]
        rank = i + 1
        print(f"#{rank:3d} | {r['job_id']:8s} | Semantic: {r['semantic_score']:.4f} | BM25: {r['bm25_score']:.4f} | Final: {r['final_score']:.4f} | {r['intitule'][:40]}", flush=True)
    print("=" * 130, flush=True)
    
    # Process results for API response
    processed_results = []
    for i, r in enumerate(results_with_bm25):
        processed_results.append({
            "job_id": r['job_id'],
            "similarity_score": round(r['semantic_score'], 2),
            "semantic_score": round(r['semantic_score'], 4),
            "bm25_score": round(r['bm25_score'], 4),
            "final_score": round(r['final_score'], 4),
            "intitule": r['intitule'],
            "entreprise": r['entreprise'],
            "lieu": r['lieu'],
            "type_contrat": r['type_contrat'],
            "date_creation": r['date_creation'],
            "matching_terms": all_matching_terms[i] if i < len(all_matching_terms) else []
        })
    
    return processed_results