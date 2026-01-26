from sentence_transformers import SentenceTransformer
from utils import search_jobs_vector
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import os
import json

device="cpu"
#384 dimensional embedding model
model = SentenceTransformer("BAAI/bge-small-en", device=device)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
translator_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

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
    """
    text = re.sub(r'\s+', ' ', text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    words = text.split()
    words = [w for w in words if w not in FRENCH_STOPWORDS]
    
    return " ".join(words).strip()

def translate_fr_to_en(text, chunk_size=400):
    """
    Split the text into parts to avoid overloading the translator
    """
    text = clean_extracted_text(text)
    
    words = text.split(' ')
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    translated_text = ""
    for chunk in chunks:
        if chunk.strip():
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            outputs = translator_model.generate(**inputs)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_text += translation + " "
            
    return translated_text.strip()

def embed_text(text):
    """
    Generates the embedding for the given text
    """
    text= translate_fr_to_en(text)
    return model.encode(text).tolist()

def embed_cv_and_search(text):
    """
    Generates the embedding for the CV and returns the top jobs from PostgreSQL.
    Also includes matching terms for each job (in English for consistency).
    """
    translated_text = translate_fr_to_en(text)
    embedding = model.encode(translated_text).tolist()
    # Pass the TRANSLATED text for consistent English term matching
    top_jobs = search_jobs_vector(embedding, cv_text=translated_text)
    return top_jobs