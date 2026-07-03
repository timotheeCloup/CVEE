import contextlib
import os
import threading
import time
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(page_title="CV Match Engine", layout="centered")

API_URL = os.getenv("API_URL", "http://localhost:8000/embed-cv")
HEALTH_URL = API_URL.rsplit("/embed-cv", 1)[0] + "/health"
COLD_START_TIMEOUT = 25

if "api_ready" not in st.session_state:
    st.session_state.api_ready = False


def _warmup_thread():
    with contextlib.suppress(Exception):
        requests.get(HEALTH_URL, timeout=60)


if not st.session_state.api_ready:
    try:
        requests.get(HEALTH_URL, timeout=3)
        st.session_state.api_ready = True
    except Exception:
        threading.Thread(target=_warmup_thread, daemon=True).start()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 5, 1])
        with col2, st.container(border=True):
            st.markdown(
                """<h3 style='text-align:center;'>Chargement...</h3>""",
                unsafe_allow_html=True,
            )
            progress_bar = st.progress(0)
            status_text = st.empty()

            for elapsed in range(1, COLD_START_TIMEOUT + 1):
                progress_bar.progress(elapsed / COLD_START_TIMEOUT)
                pct = int(elapsed / COLD_START_TIMEOUT * 100)
                status_text.markdown(
                    f"<p style='text-align:center;color:#667eea;'>{pct}%</p>",
                    unsafe_allow_html=True,
                )
                if elapsed % 5 == 1:
                    with contextlib.suppress(Exception):
                        requests.get(HEALTH_URL, timeout=2)
                time.sleep(1)

            status_text.empty()
        st.session_state.api_ready = True
        st.rerun()

st.markdown(
    """
    <style>
    .sub-title {
        font-size: 20px !important;
        color: #555;
        margin-bottom: 25px;
    }
    .company-name {
        color: #007bff;
        font-weight: 600;
        font-size: 18px;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    .stMetric {
    background-color: var(--secondary-background-color);
    color: var(--text-color);
    border: 1px solid #eee;
    padding: 10px;
    border-radius: 10px;
    }

    /* Analyse button */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%) !important;
        background-size: 200% auto !important;
        background-position: left center !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: background-position 0.5s ease, box-shadow 0.3s ease !important;
    }

    div[data-testid="stButton"] > button:hover {
        background-position: right center !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        transform: none !important;
    }

    .terms-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-top: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .term-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 6px 12px;
        border-radius: 20px;
        margin: 4px;
        font-size: 14px;
        backdrop-filter: blur(10px);
    }
    .footer {
        text-align: center;
        color: #999;
        font-size: 12px;
        padding-top: 30px;
        margin-top: 40px;
    }
    .footer-link {
        color: #999;
        text-decoration: none;
        margin: 0 8px;
        transition: color 0.2s ease;
    }
    .footer-link:hover {
        color: #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session keys
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

if "cached_job_results" not in st.session_state:
    st.session_state.cached_job_results = None


def fetch_job_results(file_bytes, file_name):
    """Get job results from the API given the CV file bytes."""
    try:
        response = requests.post(API_URL, files={"file": (file_name, file_bytes)})
        if response.status_code == 200:
            return response.json().get("top_jobs", [])
    except Exception:
        pass
    return None


st.title("📄 CV Match Engine")

st.markdown(
    '<p class="sub-title">Trouvez les offres d\'emploi qui correspondent vraiment à votre profil</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Déposez votre CV (PDF)", type=["pdf"], max_upload_size=5)

if uploaded_file is not None:
    current_upload_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if current_upload_id != st.session_state.last_upload_id:
        st.session_state.last_upload_id = current_upload_id
        st.session_state.cached_job_results = None
        for key in list(st.session_state.keys()):
            if key.startswith("analysis_"):
                del st.session_state[key]

    if st.session_state.cached_job_results is None:
        file_bytes = uploaded_file.getvalue()
        with st.spinner("Analyse du profil en cours..."):
            top_jobs = fetch_job_results(file_bytes, uploaded_file.name)
        st.session_state.cached_job_results = top_jobs
    else:
        top_jobs = st.session_state.cached_job_results

    if top_jobs:
        st.success(f"🔥 {len(top_jobs)} jobs trouvés !")

        for i, job in enumerate(top_jobs, start=1):
            raw_date = job.get("date_creation", "")
            try:
                clean_date = datetime.fromisoformat(raw_date.replace("Z", "")).strftime("%Y-%m-%d")
            except Exception:
                clean_date = "N/A"

            similarity = job.get("similarity_score", 0)
            matching_terms = job.get("matching_terms", [])
            job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job['job_id']}"
            job_key = f"analysis_{job['job_id']}"

            with st.container(border=True):
                header_col, right_col = st.columns([0.7, 0.3])

                with header_col:
                    st.markdown(f"### {i}. [{job.get('intitule', 'N/A')}]({job_url})")
                    st.markdown(
                        f'<p class="company-name">{job.get("entreprise", "N/A")}</p>',
                        unsafe_allow_html=True,
                    )

                with right_col:
                    st.metric("Match", f"{int(similarity * 100)}%")

                    if matching_terms:

                        def toggle_analysis(job_id):
                            st.session_state[job_id] = not st.session_state.get(job_id, False)

                        st.button(
                            "✨ Analyser",
                            key=f"btn_{job_key}",
                            on_click=toggle_analysis,
                            args=(job_key,),
                            use_container_width=True,
                        )

                if st.session_state.get(job_key, False) and matching_terms:
                    terms_html = "".join(
                        f'<span class="term-badge">{term}</span>' for term in matching_terms
                    )
                    st.markdown(
                        f"""
                        <div class="terms-bubble">
                            <strong>🎯 Mots-clés identifiés :</strong><br><br>
                            {terms_html}
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                c1, c2, c3 = st.columns(3)
                c1.markdown(f"📍 {job.get('lieu', 'N/A')}")
                c2.markdown(f"📄 {job.get('type_contrat', 'N/A')}")
                c3.markdown(f"📅 {clean_date}")
    else:
        st.error("Le service API n'est pas disponible. Veuillez réessayer.")

st.markdown(
    """
    <div class="footer">
        Made by Timothée Cloup-Martin<br>
        <a href="https://github.com/timotheeCloup/CVEE" target="_blank" class="footer-link">GitHub</a> •
        <a href="https://timotheecloup.github.io/portfolio/" target="_blank" class="footer-link">Portfolio</a>
    </div>
""",
    unsafe_allow_html=True,
)
