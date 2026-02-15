import pandas as pd
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
import s3fs


FT_AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
FT_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FT_SCOPE = "o2dsoffre api_offresdemploiv2"


def get_ft_token(ft_client_id, ft_client_secret):
    """get France Travail API token using client credentials"""
    try:
        response = requests.post(
            FT_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": ft_client_id,
                "client_secret": ft_client_secret,
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

def export_to_s3(jobs, aws_access_key_id, aws_secret_access_key, aws_s3_bucket_name):
    if not jobs:
        print("No jobs to export.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    aws_s3_path = f'jobs_raw/jobs_raw_{timestamp_str}.parquet'
    
    df = pd.DataFrame(jobs)
    
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: None if isinstance(x, dict) and len(x) == 0 else x
        )
    
    fs = s3fs.S3FileSystem(key=aws_access_key_id, secret=aws_secret_access_key)

    s3_uri = f"s3://{aws_s3_bucket_name}/{aws_s3_path}"
    df.to_parquet(s3_uri, engine="pyarrow", index=False, filesystem=fs)
    print(f"Export completed: {s3_uri}")


def main(ft_client_id, ft_client_secret, aws_access_key_id, aws_secret_access_key, aws_s3_bucket_name):
    """Main function to orchestrate the API to S3 pipeline"""
    token = get_ft_token(ft_client_id, ft_client_secret)
    if not token:
        raise Exception("Failed to obtain France Travail API token")
    
    jobs = fetch_jobs_data(token)
    export_to_s3(jobs, aws_access_key_id, aws_secret_access_key, aws_s3_bucket_name)


if __name__ == "__main__":
    # For local testing only
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    main(
        ft_client_id=os.getenv("FT_CLIENT_ID"),
        ft_client_secret=os.getenv("FT_CLIENT_SECRET"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_s3_bucket_name=os.getenv("AWS_S3_BUCKET_NAME"),
    )
