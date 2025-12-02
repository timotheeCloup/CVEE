import os
from dotenv import load_dotenv
import psycopg2

#Load environment variables
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT= os.getenv("DB_PORT")

# Define SQL commands
# Create tables
CREATE_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS jobs_metadata (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    title TEXT NOT NULL,
    location VARCHAR(255),
    description TEXT,
    url TEXT
);
"""

# Table for vectors
# We will use a typical dimension of 768
CREATE_VECTORS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs_vectors (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) REFERENCES jobs_metadata(job_id),
    embedding VECTOR(768) NOT NULL
);
"""

# 3. Connection and Execution
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    cur = conn.cursor()

    # Create the pgvector extension 
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Extension pgvector created.")

    # Create the tables
    cur.execute(CREATE_METADATA_TABLE)
    cur.execute(CREATE_VECTORS_TABLE)
    conn.commit()
    print("Tables 'jobs_metadata' and 'jobs_vectors' created successfully.")
    conn.close()
    
except Exception as e:
    print(f"Connection or SQL execution error: {e}")