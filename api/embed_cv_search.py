from sentence_transformers import SentenceTransformer
from utils import search_jobs_vector
from transformers import pipeline
import re

#384 dimensional embedding model
model = SentenceTransformer("BAAI/bge-small-en")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")

def clean_extracted_text(text):
    """
    Clean up text (redundant spaces etc.)
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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
            translation = translator(chunk, truncation=True)[0]['translation_text']
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
    """
    embedding = embed_text(text)
    top_jobs = search_jobs_vector(embedding)
    return top_jobs
