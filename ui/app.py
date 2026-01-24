import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="CV Match Engine", layout="centered")

st.markdown("""
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
    .stMetric { background-color: #f8f9fa; border: 1px solid #eee; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

API_URL = "http://cvee-api:8000/embed-cv"     

st.title("📄 CV Match Engine")
st.markdown('<p class="sub-title">Find the job openings that truly match your profile based on semantic analysis.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"], max_upload_size=5) 

if uploaded_file is not None:
    with st.spinner("Analyzing your profile..."):
        response = requests.post(API_URL, files={"file": uploaded_file})
        
        if response.status_code == 200:
            top_jobs = response.json().get("top_jobs", [])
            st.success(f"🔥 Found {len(top_jobs)} matching jobs!")
            
            for i, job in enumerate(top_jobs, start=1):
                raw_date = job.get('date_creation', '')
                try:
                    clean_date = datetime.fromisoformat(raw_date.replace('Z', '')).strftime('%Y-%m-%d')
                except:
                    clean_date = "N/A"

                similarity = job.get('similarity_score', 0)
                job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job['job_id']}"
                
                with st.container(border=True):
                    header_col, score_col = st.columns([0.8, 0.2])
                    
                    with header_col:
                        st.markdown(f"### {i}. [{job.get('intitule', 'N/A')}]({job_url})")
                        st.markdown(f'<p class="company-name">{job.get("entreprise", "N/A")}</p>', unsafe_allow_html=True)
                    
                    with score_col:
                        st.metric("Match", f"{int(similarity * 100)}%")
                    
                    # Détails restants sur une ligne
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"📍 {job.get('lieu', 'N/A')}")
                    c2.markdown(f"📄 {job.get('type_contrat', 'N/A')}")
                    c3.markdown(f"📅 {clean_date}")
        else:
            st.error("API unreachable. Please check your connection.")