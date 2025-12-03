import os
import pandas as pd
from dotenv import load_dotenv
import psycopg2
from psycopg2 import extras
import requests
import json

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT= os.getenv("DB_PORT")

FT_CLIENT_ID = os.getenv("FT_CLIENT_ID")
FT_CLIENT_SECRET = os.getenv("FT_CLIENT_SECRET")
FT_AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
FT_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FT_SCOPE = "o2dsoffre api_offresdemploiv2"

#authetication
def get_ft_token():
    """get France Travail API token using client credentials"""
    try:
        response = requests.post(
            FT_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": FT_CLIENT_ID,
                "client_secret": FT_CLIENT_SECRET,
                "scope": FT_SCOPE,
            },
            headers={   
                "Content-Type": "application/x-www-form-urlencoded"
            }
        )
        response.raise_for_status() # raise error for bad status codes (4xx, 5xx)
        
        token_data = response.json()
        print("France Travail API token obtained successfully.")
        return token_data.get("access_token")
    
    except Exception as e:
        print(f"Unexpected error while obtaining token: {e}")
        return None



def fetch_jobs_data(token, mots_cles="data engineer", department=None, commune=None, distance=None, domaine=None, dureeHebdo=None, experience=None, experienceExigence=None, grandDomaine=None, minCreationDate=None, niveauFormation=None, permis=None, qualification=None, region=None, salaireMin=None, secteurActivite=None, tempsPlein=None, typeContrat=None, page_size=150, max_index=3000):
    """Récupère toutes les offres d'emploi via pagination."""
    jobs = []
    start_index = 0
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    while start_index < max_index:
        params = {
            "motsCles": mots_cles,
            "range": f"{start_index}-{start_index + page_size - 1}",
            "commune": commune,
            "departement": department,
            "distance": distance,
            "domaine": domaine,
            "dureeHebdo": dureeHebdo,
            "experience": experience,
            "experienceExigence": experienceExigence,
            "grandDomaine": grandDomaine,
            "minCreationDate": minCreationDate,
            "niveauFormation": niveauFormation,
            "permis": permis,
            "qualification": qualification,
            "region": region,
            "salaireMin": salaireMin,
            "secteurActivite": secteurActivite,
            "tempsPlein": tempsPlein,
            "typeContrat": typeContrat
        }

        try:
            response = requests.get(FT_API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            jobs_page = data.get("resultats", [])
            if not jobs_page:
                print(f"No more jobs at index {start_index}.")
                break

            jobs.extend(jobs_page)
            print(f"Retrieved {len(jobs)} jobs so far.")
            start_index += page_size

        except Exception as e:
            print(f"Error at index {start_index}: {e}")
            break

    print(f"Total jobs collected: {len(jobs)}")
    return jobs

def transform_and_load(jobs_list):
    """Transform the full JSON into a DataFrame and insert all columns into PostgreSQL."""
    if not jobs_list:
        print("No jobs data to process.")
        return

    df = pd.DataFrame(jobs_list)

    df = df.where(pd.notnull(df), None)  

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()

        cols = df.columns.tolist()
        insert_query = f"""
        INSERT INTO jobs_metadata ({','.join(cols)})
        VALUES %s
        ON CONFLICT (id) DO NOTHING;
        """

        data_to_insert = []
        all_columns = set()
        for row in df.itertuples(index=False, name=None):
            all_columns.update(df.columns)
            # Convert dicts and lists to JSON strings
            row_json = tuple(json.dumps(x) if isinstance(x, (dict, list)) else x for x in row)
            data_to_insert.append(row_json)

        print("Colonnes maximales détectées :", all_columns, len(all_columns))
        
        print(f"-> Attempting to insert {len(data_to_insert)} rows...")
        #insert data in one batch
        extras.execute_values(cur, insert_query, data_to_insert)
        conn.commit()
        print("-> Data successfully inserted/updated in 'jobs_metadata'.")
    except Exception as e:
        print(f"Error during insertion: {e}")
        if 'conn' in locals() and conn:
            conn.rollback()
    finally:
        if 'conn' in locals() and conn:
            cur.close()
            conn.close()




def test_save_jobs_to_json():
    """Save jobs data to a JSON file."""
    token = get_ft_token()
    jobs = fetch_jobs_data(token)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "sandbox")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "jobs.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(jobs)} jobs to {output_path}")




if __name__ == "__main__":
    token = get_ft_token()
    if token:
        jobs_data = fetch_jobs_data(token, mots_cles="data scientist")
        transform_and_load(jobs_data)