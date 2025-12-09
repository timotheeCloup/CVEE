import os
import pandas as pd
from dotenv import load_dotenv
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import s3fs

load_dotenv()


timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

FT_CLIENT_ID = os.getenv("FT_CLIENT_ID")
FT_CLIENT_SECRET = os.getenv("FT_CLIENT_SECRET")
FT_AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
FT_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FT_SCOPE = "o2dsoffre api_offresdemploiv2"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
AWS_S3_PATH=f'jobs_raw/jobs_raw_{timestamp_str}.parquet'

#authentication
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



def fetch_jobs_data(token, page_size=150, max_index=3000, sort=1, niveauFormation="NV2", publieeDepuis=1, mots_cles=None, department=None, commune=None, distance=None, domaine=None, dureeHebdo=None, experience=None, experienceExigence=None, grandDomaine=None, minCreationDate=None, maxCreationDate=None, permis=None, qualification=None, region=None, salaireMin=None, secteurActivite=None, tempsPlein=None, typeContrat=None):
    """Fetch job data from France Travail API with pagination"""
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
            "sort": sort,
            "commune": commune,
            "departement": department,
            "distance": distance,
            "domaine": domaine,
            "dureeHebdo": dureeHebdo,
            "experience": experience,
            "experienceExigence": experienceExigence,
            "grandDomaine": grandDomaine,
            "minCreationDate": minCreationDate,
            "maxCreationDate": maxCreationDate,
            "niveauFormation": niveauFormation, # NV2 : Bac+5 et plus ou équivalents
            "permis": permis,
            "qualification": qualification,
            "region": region,
            "salaireMin": salaireMin,
            "secteurActivite": secteurActivite,
            "tempsPlein": tempsPlein,
            "typeContrat": typeContrat,
            "publieeDepuis": publieeDepuis # if x>1 --> doesn't work 
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

def export_to_s3(jobs):
    if not jobs:
        print("No jobs to export.")
        return

    df = pd.DataFrame(jobs)
    fs = s3fs.S3FileSystem(key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)

    s3_uri = f"s3://{AWS_S3_BUCKET_NAME}/{AWS_S3_PATH}"
    df.to_parquet(s3_uri, engine="pyarrow", index=False, filesystem=fs)
    print(f"Export completed: {s3_uri}")


def main():
    token = get_ft_token()
    if not token:
        return
    jobs = fetch_jobs_data(token)
    export_to_s3(jobs)

if __name__ == "__main__":
    main()