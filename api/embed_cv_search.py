from sentence_transformers import SentenceTransformer
from utils import search_jobs_vector

#384 dimensional embedding model
model = SentenceTransformer("BAAI/bge-small-en")

def embed_text(text):
    """
    Generates the embedding for the given text
    """
    return model.encode(text).tolist()

def embed_cv_and_search(text):
    """
    Generates the embedding for the CV and returns the top jobs from PostgreSQL.
    """
    embedding = embed_text(text)
    top_jobs = search_jobs_vector(embedding)
    return top_jobs
