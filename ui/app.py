# ui/app.py
import streamlit as st
import requests

API_URL = "http://cvee-api:8000/embed-cv"     
#API_URL = "http://cvee-api.cvee.svc.cluster.local:8000/embed-cv" # Alternative internal URL

st.set_page_config(page_title="CV-Embedding Engine", layout="centered")

st.title("CV-Embedding Engine")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing your CV..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files={"file": uploaded_file})
        
        if response.status_code == 200:
            top_jobs = response.json().get("top_jobs", [])
            st.success(f"Found {len(top_jobs)} matching jobs!")
            
            for i, job in enumerate(top_jobs, start=1):
                job_id = job['job_id']
                job_url = f"https://candidat.francetravail.fr/offres/recherche/detail/{job_id}"
                st.markdown(f"**{i}. [Job ID: {job_id}]({job_url})**")
        else:
            st.error("Error processing CV. Make sure the API is running.")
