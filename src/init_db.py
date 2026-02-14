import os
from dotenv import load_dotenv
import psycopg2


load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT= os.getenv("DB_PORT", 5432)

# Define SQL commands
# Create tables
CREATE_SILVER_TABLE = """
CREATE TABLE IF NOT EXISTS jobs_silver (
    job_id TEXT PRIMARY KEY,

    intitule TEXT,
    description TEXT,
    vector_text_input TEXT,
    dateCreation TEXT,
    dateActualisation TEXT,

    lieuTravail JSONB,
    entreprise JSONB,
    contact JSONB,
    agence JSONB,
    origineOffre JSONB,
    contexteTravail JSONB,
    salaire JSONB,

    competences JSONB,
    formations JSONB,
    langues JSONB,
    qualitesProfessionnelles JSONB,
    permis JSONB,

    romeCode TEXT,
    romeLibelle TEXT,
    appellationlibelle TEXT,
    typeContrat TEXT,
    typeContratLibelle TEXT,
    natureContrat TEXT,
    experienceExige TEXT,
    experienceLibelle TEXT,
    dureeTravailLibelle TEXT,
    dureeTravailLibelleConverti TEXT,
    alternance BOOLEAN,
    nombrePostes INTEGER, 
    accessibleTH BOOLEAN,
    qualificationCode TEXT,
    qualificationLibelle TEXT,
    codeNAF TEXT,
    secteurActivite TEXT,
    secteurActiviteLibelle TEXT,
    trancheEffectifEtab TEXT,
    offresManqueCandidats BOOLEAN,
    entrepriseAdaptee BOOLEAN,
    employeurHandiEngage BOOLEAN,
    deplacementCode TEXT,
    deplacementLibelle TEXT,
    experienceCommentaire TEXT,
    complementExercice TEXT,
    ingestion_date DATE
);
"""

# Table for vectors
# We will use a typical dimension of 384
CREATE_GOLD_TABLE = """
CREATE TABLE IF NOT EXISTS jobs_gold (
    job_id TEXT PRIMARY KEY,
    embedding vector(384),
    --foreign key constraint
    CONSTRAINT fk_job
        FOREIGN KEY (job_id)
        REFERENCES jobs_silver(job_id)
        ON DELETE CASCADE
);
"""

# 3. Connection and Execution
try:
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
    )
    cur = conn.cursor()

    # Create the pgvector extension 
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    print("Extension pgvector created.")

    # Create the tables
    cur.execute("DROP TABLE IF EXISTS jobs_silver CASCADE;")
    cur.execute("DROP TABLE IF EXISTS jobs_gold CASCADE;") 
    cur.execute(CREATE_SILVER_TABLE)
    cur.execute(CREATE_GOLD_TABLE)
    conn.commit()
    print("Tables 'jobs_silver' and 'jobs_gold' created successfully.")
    conn.close()
    
except Exception as e:
    print(f"Connection or SQL execution error: {e}")