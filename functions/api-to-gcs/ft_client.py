from datetime import datetime

import gcsfs
import pandas as pd
import requests

FT_AUTH_URL = "https://entreprise.francetravail.fr/connexion/oauth2/access_token?realm=/partenaire"
FT_API_URL = "https://api.francetravail.io/partenaire/offresdemploi/v2/offres/search"
FT_SCOPE = "o2dsoffre api_offresdemploiv2"

DEFAULT_PAGE_SIZE = 150
DEFAULT_MAX_INDEX = 3000


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
            timeout=30,
        )
        response.raise_for_status()
        token_data = response.json()
        print("France Travail API token obtained successfully.")
        return token_data.get("access_token")
    except Exception as e:
        print(f"Error obtaining token: {e}")
        return None


def _build_params(
    start_index, page_size, sort, niveau_formation, publiee_depuis, min_date, max_date
):
    params = {
        "range": f"{start_index}-{start_index + page_size - 1}",
        "sort": sort,
        "niveauFormation": niveau_formation,
    }
    if min_date and max_date:
        params["minDateCreation"] = min_date
        params["maxDateCreation"] = max_date
    else:
        params["publieeDepuis"] = publiee_depuis
    return params


def fetch_jobs_data(
    token,
    page_size=DEFAULT_PAGE_SIZE,
    max_index=DEFAULT_MAX_INDEX,
    sort=1,
    niveau_formation="NV1",
    publiee_depuis=1,
    date_min=None,
    date_max=None,
    max_results=None,
):
    """
    Fetch job data from France Travail API with pagination.

    Two modes:
    - Daily mode (default): uses `publieeDepuis` (last N days)
    - Historical backfill: uses `date_min`/`date_max` (YYYY-MM-DD range)

    max_results overrides max_index to limit total jobs fetched.
    """
    jobs = []
    start_index = 0
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if max_results is not None:
        max_index = max_results

    while start_index < max_index:
        params = _build_params(
            start_index,
            page_size,
            sort,
            niveau_formation,
            publiee_depuis,
            date_min,
            date_max,
        )
        try:
            response = requests.get(FT_API_URL, headers=headers, params=params, timeout=60)
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


def _make_filename(date_min, date_max):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if date_min and date_max:
        return f"jobs_raw_{date_min}_{date_max}_{ts}.parquet"
    return f"jobs_raw_{ts}.parquet"


def export_to_gcs(jobs, bucket_name, date_min=None, date_max=None):
    """Export jobs DataFrame to GCS as Parquet using Application Default Credentials"""
    if not jobs:
        print("No jobs to export.")
        return

    filename = _make_filename(date_min, date_max)
    gcs_path = f"gs://{bucket_name}/jobs_raw/{filename}"

    df = pd.DataFrame(jobs)

    for col in df.columns:
        df[col] = df[col].apply(lambda x: None if isinstance(x, dict) and len(x) == 0 else x)

    fs = gcsfs.GCSFileSystem()
    df.to_parquet(gcs_path, engine="pyarrow", index=False, filesystem=fs)
    print(f"Export completed: {gcs_path}")


def main(
    ft_client_id,
    ft_client_secret,
    bucket_name,
    date_min=None,
    date_max=None,
    publiee_depuis=1,
    max_results=None,
):
    """
    Fetch jobs from FT API → export to GCS.

    Daily mode (default):       main(..., publiee_depuis=1)      → jobs from last 24h
    Historical backfill:        main(..., date_min="2026-01-01", date_max="2026-01-31")
    Limit results:              main(..., max_results=100)
    """
    token = get_ft_token(ft_client_id, ft_client_secret)
    if not token:
        raise RuntimeError("Failed to obtain France Travail API token")

    jobs = fetch_jobs_data(
        token,
        date_min=date_min,
        date_max=date_max,
        publiee_depuis=publiee_depuis,
        max_results=max_results,
    )
    export_to_gcs(jobs, bucket_name, date_min=date_min, date_max=date_max)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    main(
        ft_client_id=os.getenv("FT_CLIENT_ID"),
        ft_client_secret=os.getenv("FT_CLIENT_SECRET"),
        bucket_name=os.getenv("GCS_BUCKET_NAME"),
    )
