from sentence_transformers import SentenceTransformer
from utils import search_jobs_vector
from google.cloud import translate_v2
import re
import os
import json
import time

device="cpu"
#384 dimensional embedding model
model = SentenceTransformer("BAAI/bge-small-en", device=device)
translate_client = translate_v2.Client()

def load_french_stopwords():
    path = os.path.join(os.path.dirname(__file__), 'stopwords.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('fr', []))
    except Exception:
        return set()


FRENCH_STOPWORDS = load_french_stopwords()

def clean_extracted_text(text):
    """
    Clean up text (remove stopwords, redundant spaces et.)
    Preserves useful punctuation: +, #, - (for C++, C#, full-stack, etc.)
    """
    text = re.sub(r'\s+', ' ', text).lower()
    # Keep useful punctuation: +, #, -
    text = re.sub(r'[^\w\s\+\#\-]', ' ', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    words = text.split()
    words = [w for w in words if w not in FRENCH_STOPWORDS and len(w) > 1]
    
    return " ".join(words).strip()

def translate_fr_to_en(text, chunk_size=5000):
    """
    Translate French text to English using Google Cloud Translation API.
    Much faster and lighter than Hugging Face models.
    """
    text = clean_extracted_text(text)
    
    if not text.strip():
        return ""
    
    # Split into chunks for API limits (Google allows up to 100k chars per request)
    words = text.split(' ')
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    translated_chunks = []
    for chunk in chunks:
        if chunk.strip():
            try:
                result = translate_client.translate(
                    chunk,
                    target_language='en'
                )
                translated_chunks.append(result['translatedText'])
            except Exception as e:
                print(f"Translation error: {str(e)}", flush=True)
                # Fallback: return original text if translation fails
                translated_chunks.append(chunk)
    
    return " ".join(translated_chunks).strip()


def embed_cv_and_search(text, t_api_start=None):
    """
    Generates the embedding for the CV and returns the top jobs from PostgreSQL.
    Also includes matching terms for each job (in English for consistency).
    
    Args:
        text: CV text to process
        t_api_start: Start time of API call (for accurate timing)
    """
    if t_api_start is None:
        t_api_start = time.time()
    
    t0 = time.time()
    
    translated_text = translate_fr_to_en(text)
    t1 = time.time()
    print(f"\n=== TIMING BREAKDOWN ===", flush=True)
    print(f"0. API call to processing: {t0 - t_api_start:7.2f}s", flush=True)
    print(f"1. Translation:            {t1 - t0:7.2f}s", flush=True)
    print(f"   Text length:            {len(translated_text)} chars", flush=True)
    
    embedding = model.encode(translated_text).tolist()
    t2 = time.time()
    print(f"2. Embedding:              {t2 - t1:7.2f}s", flush=True)
    print(f"   Embedding dim:          {len(embedding)}", flush=True)
    
    # Pass the TRANSLATED text for consistent English term matching
    top_jobs = search_jobs_vector(embedding, cv_text=translated_text)
    t3 = time.time()
    print(f"3. DB Search + FTS:        {t3 - t2:7.2f}s", flush=True)
    print(f"   Results found:          {len(top_jobs)} jobs", flush=True)
    print(f"\n4. TOTAL TIME:             {t3 - t_api_start:7.2f}s", flush=True)
    print(f"=" * 35, flush=True)
    
    return top_jobs