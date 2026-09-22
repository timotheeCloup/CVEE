import contextlib
import json
import os
import threading
import time
from datetime import datetime

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="CV Match Engine", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000/embed-cv")
HEALTH_URL = API_URL.rsplit("/embed-cv", 1)[0] + "/health"
COLD_START_TIMEOUT = 10
DEPARTEMENTS_FILE = os.path.join(os.path.dirname(__file__), "departements.json")

# When true, the API is a private Cloud Run service and requests must carry an
# identity token (audience = API base URL). The token is fetched from the
# metadata server, so this only works from inside Cloud Run.
API_AUTH = os.getenv("API_AUTH", "").lower() in ("1", "true", "yes")
_API_AUDIENCE = API_URL.rsplit("/embed-cv", 1)[0]
_METADATA_IDENTITY_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity"
)


def api_headers() -> dict:
    """Return auth headers for the API, or {} when the API is public."""
    if not API_AUTH:
        return {}
    try:
        response = requests.get(
            _METADATA_IDENTITY_URL,
            params={"audience": _API_AUDIENCE},
            headers={"Metadata-Flavor": "Google"},
            timeout=2,
        )
        response.raise_for_status()
        return {"Authorization": f"Bearer {response.text}"}
    except Exception:
        return {}


@st.cache_data
def load_departements() -> dict[str, str]:
    """Load the static French department code -> name mapping."""
    with open(DEPARTEMENTS_FILE, encoding="utf-8") as f:
        return json.load(f)


DEPARTEMENTS = load_departements()
CONTRAT_TYPES = ["CDI", "CDD"]

if "api_ready" not in st.session_state:
    st.session_state.api_ready = False


def _warmup_thread():
    with contextlib.suppress(Exception):
        requests.get(HEALTH_URL, timeout=60, headers=api_headers())


if not st.session_state.api_ready:
    try:
        requests.get(HEALTH_URL, timeout=3, headers=api_headers())
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
                # ONNX runtime boots in a few seconds (no torch load), so poll
                # health every second and stop as soon as the API answers
                # instead of always waiting the full timeout.
                ready = False
                with contextlib.suppress(Exception):
                    requests.get(HEALTH_URL, timeout=2, headers=api_headers())
                    ready = True
                if ready:
                    progress_bar.progress(1.0)
                    break
                progress_bar.progress(elapsed / COLD_START_TIMEOUT)
                pct = int(elapsed / COLD_START_TIMEOUT * 100)
                status_text.markdown(
                    f"<p style='text-align:center;color:#667eea;'>{pct}%</p>",
                    unsafe_allow_html=True,
                )
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

if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

if "show_map" not in st.session_state:
    st.session_state.show_map = False

if "scrolled_job" not in st.session_state:
    st.session_state.scrolled_job = None

# Layout: centered single column by default, offers/map split when the map is on.
if st.session_state.show_map:
    st.markdown(
        """
        <style>
        /* Keep the map vertically centered while the offers scroll past it. */
        div[data-testid="stColumn"]:has(div[data-testid="stDeckGlJsonChart"]) {
            position: sticky;
            top: max(1rem, calc(50vh - 260px));
            align-self: flex-start;
            height: fit-content;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] { max-width: 780px; margin: 0 auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def fetch_job_results(file_bytes, file_name, departements=None, types_contrat=None):
    """Get job results from the API given the CV file bytes and optional filters."""
    data = {}
    if departements:
        data["departements"] = ",".join(departements)
    if types_contrat:
        data["types_contrat"] = ",".join(types_contrat)
    try:
        response = requests.post(
            API_URL,
            files={"file": (file_name, file_bytes)},
            data=data or None,
            headers=api_headers(),
        )
        if response.status_code == 200:
            return response.json().get("top_jobs", [])
    except Exception:
        pass
    return None


def _score_color(normalized: float) -> list[int]:
    """Map a normalized score (0..1) to an RGBA color: blue (low) -> violet (high).

    Reuses the "Analyser" button gradient (#667eea -> #764ba2).
    """
    low, high = (102, 126, 234), (118, 75, 162)
    return [round(low[i] + (high[i] - low[i]) * normalized) for i in range(3)] + [230]


def render_offers_map(jobs: list[dict]) -> str | None:
    """Plot the matching offers on a map, one point per geolocated offer.

    Offers without coordinates (country/region-level) are skipped. Pin color and
    size reflect the match score. The radius is expressed in meters (so it grows
    as you zoom in) but clamped between 4 and 12 screen pixels: past that cap a
    cluster of offers stays distinguishable instead of merging into one blob.

    Returns the id of the offer whose pin was clicked (if any).
    """
    points = [
        {
            "job_id": job["job_id"],
            "intitule": job.get("intitule") or "N/A",
            "entreprise": job.get("entreprise") or "N/A",
            "lieu": job.get("lieu") or "N/A",
            "match": int(job.get("similarity_score", 0) * 100),
            "url": f"https://candidat.francetravail.fr/offres/recherche/detail/{job['job_id']}",
            "latitude": job["latitude"],
            "longitude": job["longitude"],
        }
        for job in jobs
        if job.get("latitude") is not None and job.get("longitude") is not None
    ]
    if not points:
        return None

    df = pd.DataFrame(points)
    scores = df["match"].astype(float)
    score_min, score_max = float(scores.min()), float(scores.max())
    span = score_max - score_min
    # Normalize the score across the displayed offers (blue = worst, violet = best).
    normalized = (scores - score_min) / span if span > 0 else pd.Series(1.0, index=scores.index)
    df["color"] = [_score_color(float(t)) for t in normalized]
    df["radius"] = 9000 + scores * 120

    layer = pdk.Layer(
        "ScatterplotLayer",
        id="jobs",
        data=df,
        get_position="[longitude, latitude]",
        get_radius="radius",
        get_fill_color="color",
        radius_min_pixels=3,
        radius_max_pixels=10,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": (
            "<b>{intitule}</b><br/>{entreprise}<br/>📍 {lieu}<br/>Match : {match}%<br/>"
            "<a href='{url}' target='_blank'>Voir l'offre ↗</a>"
        ),
        "style": {"backgroundColor": "#667eea", "color": "white"},
    }

    event = st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(latitude=46.6, longitude=2.4, zoom=5),
            map_style=pdk.map_styles.CARTO_LIGHT,
            tooltip=tooltip,
        ),
        width="stretch",
        height=520,
        key="offers_map",
        on_select="rerun",
        selection_mode="single-object",
    )

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#888;margin-top:6px;">
          <span>{int(score_min)}%</span>
          <div style="flex:1;height:8px;border-radius:4px;
                      background:linear-gradient(90deg,#667eea,#764ba2);"></div>
          <span>{int(score_max)}%</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = len(jobs) - len(points)
    if missing:
        st.caption(f"{missing} offre(s) sans localisation ne figurent pas sur la carte.")

    try:
        objects = event["selection"]["objects"].get("jobs") or []
    except (KeyError, TypeError):
        return None
    return objects[0].get("job_id") if objects else None


def render_feed(jobs: list[dict]) -> None:
    """Render the ranked offer cards."""
    for i, job in enumerate(jobs, start=1):
        raw_date = job.get("date_creation", "")
        try:
            clean_date = datetime.fromisoformat(raw_date.replace("Z", "")).strftime("%Y-%m-%d")
        except Exception:
            clean_date = "N/A"

        similarity = job.get("similarity_score", 0)
        matching_terms = job.get("matching_terms", [])
        job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job['job_id']}"
        job_key = f"analysis_{job['job_id']}"

        with st.container(border=True, key=f"job-{job['job_id']}"):
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


def scroll_to_offer(job_id: str | None) -> None:
    """Scroll the page to the clicked offer's card, centered vertically.

    Runs a tiny script in the parent document: Streamlit has no native API to
    scroll to an element, and the pin click only gives us the offer id.
    """
    if not job_id:
        st.session_state.scrolled_job = None
        return
    if job_id == st.session_state.scrolled_job:
        return
    st.session_state.scrolled_job = job_id
    components.html(
        f"""
        <script>
        setTimeout(() => {{
            const el = window.parent.document.querySelector('.st-key-job-{job_id}');
            if (el) el.scrollIntoView({{behavior: "smooth", block: "center"}});
        }}, 300);
        </script>
        """,
        height=1,
    )


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
        st.session_state.results_cache = {}
        for key in list(st.session_state.keys()):
            if key.startswith("analysis_"):
                del st.session_state[key]

    with st.form("filters_form"):
        selected_departements = st.multiselect(
            "Département",
            options=list(DEPARTEMENTS),
            format_func=lambda code: f"{code} - {DEPARTEMENTS[code]}",
            placeholder="Tous les départements",
        )
        selected_types = st.multiselect(
            "Type de contrat",
            options=CONTRAT_TYPES,
            placeholder="Tous les contrats",
        )
        apply_col, map_col = st.columns([0.75, 0.25])
        with apply_col:
            st.form_submit_button("Appliquer les filtres")
        with map_col:
            if st.form_submit_button("Carte", width="stretch"):
                st.session_state.show_map = not st.session_state.show_map
                st.rerun()

    cache_key = (
        current_upload_id,
        tuple(sorted(selected_departements)),
        tuple(sorted(selected_types)),
    )
    cache = st.session_state.results_cache
    if cache_key not in cache:
        file_bytes = uploaded_file.getvalue()
        with st.spinner("Analyse du profil en cours..."):
            cache[cache_key] = fetch_job_results(
                file_bytes,
                uploaded_file.name,
                selected_departements,
                selected_types,
            )
    top_jobs = cache[cache_key]

    if top_jobs:
        st.success(f"🔥 {len(top_jobs)} jobs trouvés !")

        if st.session_state.show_map:
            feed_col, map_col = st.columns([0.62, 0.38], gap="large")
            with feed_col:
                render_feed(top_jobs)
            with map_col:
                scroll_to_offer(render_offers_map(top_jobs))
        else:
            render_feed(top_jobs)
    elif top_jobs is None:
        st.error("Le service API n'est pas disponible. Veuillez réessayer.")
    else:
        st.info("Aucune offre ne correspond à ces filtres.")

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
