"""
CARA — Cardiovascular AI Risk Advisor Clinical Dashboard

Streamlit Application Setup Instructions:
1. Install dependencies:
   pip install streamlit joblib faiss-cpu sentence-transformers google-generativeai pandas numpy
2. Place these files in the same folder as this script:
   - lr_model.joblib
   - scaler.joblib
   - feature_names.joblib
   - faiss.index
   - mtsamples.csv
3. Run the app:
   streamlit run cara_app.py
"""

import streamlit as st
import joblib
import faiss
import numpy as np
import pandas as pd
import json
import time
from sentence_transformers import SentenceTransformer
import google.generativeai as genai


def get_gemini_api_key() -> str | None:
    """Read Gemini API key from Streamlit secrets."""
    return st.secrets.get("GEMINI_API_KEY")


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CARA — Cardiovascular AI Risk Advisor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
""", unsafe_allow_html=True)

# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all model artifacts once and cache them."""
    lr_model = joblib.load("lr_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    faiss_index = faiss.read_index("faiss.index")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return lr_model, scaler, feature_names, faiss_index, embed_model


@st.cache_data
def load_mtsamples():
    """Load and filter MTSamples knowledge base."""
    df = pd.read_csv("mtsamples.csv")
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    relevant = [
        "Cardiovascular / Pulmonary",
        "General Medicine",
        "Consult - History and Phy.",
        "Discharge Summary",
        "SOAP / Chart / Progress Notes",
        "Emergency Room Reports",
        "Nephrology"
    ]
    df = df[df["medical_specialty"].isin(relevant)].dropna(subset=["transcription"])
    return df.reset_index(drop=True)


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def build_clinical_query(features: dict) -> str:
    """Convert patient features to natural language search query."""
    parts = [
        f"age {int(features.get('age', 55))} years",
        f"resting blood pressure {int(features.get('trestbps', 130))} mmHg",
        f"cholesterol {int(features.get('chol', 240))} mg/dL",
        f"maximum heart rate {int(features.get('thalch', 150))} bpm",
        f"ST depression {features.get('oldpeak', 1.0):.1f}",
    ]
    return "cardiac patient: " + ", ".join(parts) + ". Cardiovascular risk assessment, chest pain, coronary artery disease."


def retrieve_notes(query: str, faiss_index, embed_model, df_rag, k: int = 3):
    """Search FAISS for top-k most relevant clinical notes."""
    query_emb = embed_model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_emb, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "specialty": df_rag.iloc[idx]["medical_specialty"],
            "description": df_rag.iloc[idx]["description"],
            "text": df_rag.iloc[idx]["transcription"][:600],
            "distance": float(dist)
        })
    return results


def build_prompt(risk_score: float, risk_label: str, features: dict, notes: list) -> str:
    """Build dual-audience Gemini prompt."""
    notes_text = ""
    for i, note in enumerate(notes):
        notes_text += f"\n[Note {i+1} — {note['specialty']}]\n{note['text']}\n"

    features_str = ", ".join([f"{k}: {v}" for k, v in features.items()])

    return f"""
ROLE: You are simultaneously a senior cardiologist and a patient health educator.

PATIENT CLINICAL PROFILE:
- Risk Score: {risk_score:.4f} (0.0=no risk, 1.0=highest risk)
- Risk Classification: {risk_label}
- Clinical Features: {features_str}

RETRIEVED CLINICAL EVIDENCE:
{notes_text}

CHAIN-OF-THOUGHT:
Before writing, reason through:
1. Which features are driving the risk score?
2. What do the retrieved notes say about similar cases?
3. What are the most important clinical actions?

FEW-SHOT FORMAT — follow exactly:
---
CLINICAL SUMMARY (For the Doctor):
Patient presents with [key features]. Risk of [X] indicates [level]. Primary drivers: [features]. Evidence suggests [finding]. Next steps: [actions].

PATIENT EXPLANATION (For the Patient):
Your heart check shows [simple finding]. Think of [analogy]. Main concerns: [simplified]. You can help by [actions].
---

CONSTRAINTS:
- CLINICAL: Medical terminology, specific values, precise.
- PATIENT: No jargon, everyday analogies, honest but reassuring.
- BOTH: Ground ONLY in retrieved notes. Do NOT fabricate.
- Keep each to 4-6 sentences maximum.

Generate both explanations now:
"""


def generate_explanation(prompt: str, api_key: str | None = None) -> str:
    """Call Gemini API to generate dual-audience explanation."""
    api_key = api_key or get_gemini_api_key()
    if not api_key:
        raise ValueError("Gemini API key not found. Add GEMINI_API_KEY in Streamlit secrets.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    response = model.generate_content(prompt)
    return response.text


def get_risk_color(score: float):
    if score >= 0.5:
        return "#E74C3C", "HIGH RISK", "risk-high"
    else:
        return "#27AE60", "LOW RISK", "risk-low"


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""

Powered by Logistic Regression + RAG + Gemini 2.5 Flash-Lite | Heart Disease UCI Dataset | MTSamples Knowledge Base

Features entered via sidebar controls

Features with highest odds ratios from Logistic Regression model

Top 3 most relevant notes from MTSamples knowledge base (1,576 cardiovascular notes)

Generated by Gemini 2.5 Flash-Lite — grounded in retrieved clinical evidence
""")
