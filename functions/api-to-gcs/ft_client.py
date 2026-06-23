from datetime import datetime

import gcsfs
import pandas as pd
import requests

FT_AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
FT_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FT_SCOPE = "o2dsoffre api_offresdemploiv2"


def get_ft_token(ft_client_id, ft_client_secret):
    """Obtain France Travail API OAuth2 token"""
    try:
        response = requests.post(
            FT_AUTH_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": ft_client_id,
                "client_secret": ft_client_secret,
                "scope": FT_SCOPE,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        token_data = response.json()
        print("France Travail API token obtained successfully.")
        return token_data.get("access_token")
    except Exception as e:
        print(f"Error obtaining token: {e}")
        return None


def fetch_jobs_data(
    token, page_size=150, max_index=3000, sort=1, niveauFormation="NV2", publieeDepuis=1, **kwargs
):
    """Fetch job data from France Travail API with pagination"""
    jobs = []
    start_index = 0
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    while start_index < max_index:
        params = {
            "range": f"{start_index}-{start_index + page_size - 1}",
            "sort": sort,
            "niveauFormation": niveauFormation,
            "publieeDepuis": publieeDepuis,
            **kwargs,
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


def export_to_gcs(jobs, bucket_name):
    """Export jobs DataFrame to GCS as Parquet using Application Default Credentials"""
    if not jobs:
        print("No jobs to export.")
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_path = f"gs://{bucket_name}/jobs_raw/jobs_raw_{timestamp_str}.parquet"

    df = pd.DataFrame(jobs)

    # Replace empty dicts with None (Parquet compatibility)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: None if isinstance(x, dict) and len(x) == 0 else x)

    fs = gcsfs.GCSFileSystem()
    df.to_parquet(gcs_path, engine="pyarrow", index=False, filesystem=fs)
    print(f"Export completed: {gcs_path}")


def main(ft_client_id, ft_client_secret, bucket_name):
    """Main function: fetch jobs from FT API → export to GCS"""
    token = get_ft_token(ft_client_id, ft_client_secret)
    if not token:
        raise RuntimeError("Failed to obtain France Travail API token")

    jobs = fetch_jobs_data(token)
    export_to_gcs(jobs, bucket_name)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(
        ft_client_id=os.getenv("FT_CLIENT_ID"),
        ft_client_secret=os.getenv("FT_CLIENT_SECRET"),
        bucket_name=os.getenv("GCS_BUCKET_NAME"),
    )
