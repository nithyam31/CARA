"""
CARA — Cardiovascular AI Risk Advisor
Clinical Dashboard — Streamlit Application

Setup Instructions:
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

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CARA — Cardiovascular AI Risk Advisor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main background */
    .stApp {
        background: #0A1628;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0D1F3C;
        border-right: 1px solid #1B3A6B;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #48C9B0;
    }

    /* Slider labels */
    .stSlider label {
        color: #A8C5DA !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* Slider track */
    .stSlider [data-baseweb="slider"] {
        margin-top: 4px;
    }

    /* Select box */
    .stSelectbox label {
        color: #A8C5DA !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }

    /* Text input */
    .stTextInput label {
        color: #A8C5DA !important;
    }

    /* Main content area */
    .main .block-container {
        padding: 1.5rem 2rem;
        max-width: 1400px;
    }

    /* CARA header */
    .cara-header {
        background: linear-gradient(135deg, #0D1F3C 0%, #1B3A6B 50%, #0D1F3C 100%);
        border: 1px solid #1B6CA8;
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .cara-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #48C9B0, #1B6CA8, #48C9B0);
    }

    .cara-title {
        font-size: 32px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .cara-subtitle {
        font-size: 14px;
        color: #48C9B0;
        margin: 4px 0 0 0;
        font-weight: 400;
    }

    .cara-badge {
        background: #48C9B0;
        color: #0A1628;
        font-size: 11px;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }

    /* Risk score card */
    .risk-card {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }

    .risk-high {
        background: linear-gradient(135deg, #2D0A0A, #4A1010);
        border-color: #E74C3C;
    }

    .risk-low {
        background: linear-gradient(135deg, #0A2D1A, #0F4A2A);
        border-color: #27AE60;
    }

    .risk-borderline {
        background: linear-gradient(135deg, #2D1F0A, #4A3510);
        border-color: #F39C12;
    }

    .risk-score-number {
        font-size: 64px;
        font-weight: 700;
        line-height: 1;
        margin: 8px 0;
        font-family: 'DM Mono', monospace;
    }

    .risk-label {
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .risk-description {
        font-size: 12px;
        opacity: 0.7;
        margin-top: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: #0D1F3C;
        border: 1px solid #1B3A6B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #48C9B0;
        font-family: 'DM Mono', monospace;
    }

    .metric-label {
        font-size: 11px;
        color: #7F9DB0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        background: #0D1F3C;
        border-left: 3px solid #1B6CA8;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        margin: 20px 0 12px 0;
    }

    .section-header h3 {
        color: #FFFFFF;
        font-size: 14px;
        font-weight: 600;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .section-header p {
        color: #7F9DB0;
        font-size: 12px;
        margin: 2px 0 0 0;
    }

    /* Clinical note card */
    .note-card {
        background: #0D1F3C;
        border: 1px solid #1B3A6B;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        position: relative;
    }

    .note-rank {
        display: inline-block;
        background: #1B6CA8;
        color: white;
        font-size: 10px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
        margin-bottom: 8px;
        font-family: 'DM Mono', monospace;
    }

    .note-specialty {
        font-size: 11px;
        color: #48C9B0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .note-description {
        font-size: 13px;
        color: #A8C5DA;
        font-weight: 500;
        margin: 4px 0;
    }

    .note-text {
        font-size: 12px;
        color: #7F9DB0;
        line-height: 1.6;
        font-family: 'DM Mono', monospace;
        background: #061020;
        padding: 12px;
        border-radius: 8px;
        margin-top: 8px;
        border: 1px solid #0D2040;
    }

    /* Explanation boxes */
    .explanation-clinical {
        background: linear-gradient(135deg, #0D1F3C, #142840);
        border: 1px solid #1B6CA8;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .explanation-patient {
        background: linear-gradient(135deg, #0A2D1A, #0F3520);
        border: 1px solid #27AE60;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .explanation-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .explanation-label-clinical { color: #1B6CA8; }
    .explanation-label-patient { color: #27AE60; }

    .explanation-text {
        font-size: 14px;
        line-height: 1.8;
        color: #D4E8F4;
    }

    /* Odds ratio bar */
    .odds-bar-container {
        margin-bottom: 10px;
    }

    .odds-bar-label {
        font-size: 12px;
        color: #A8C5DA;
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #48C9B0;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #1B6CA8, #48C9B0);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 32px;
        font-size: 14px;
        font-weight: 600;
        width: 100%;
        letter-spacing: 0.5px;
        transition: all 0.2s;
        font-family: 'DM Sans', sans-serif;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(72, 201, 176, 0.3);
    }

    /* Divider */
    hr {
        border-color: #1B3A6B;
        margin: 20px 0;
    }

    /* Warning/info boxes */
    .stWarning, .stInfo, .stSuccess, .stError {
        border-radius: 10px;
    }

    /* Feature tag */
    .feature-tag {
        display: inline-block;
        background: #061020;
        border: 1px solid #1B3A6B;
        color: #48C9B0;
        font-size: 11px;
        font-family: 'DM Mono', monospace;
        padding: 3px 8px;
        border-radius: 4px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODELS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all model artifacts once and cache them."""
    lr_model      = joblib.load("lr_model.joblib")
    scaler        = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    faiss_index   = faiss.read_index("faiss.index")
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
    return lr_model, scaler, feature_names, faiss_index, embed_model

@st.cache_data
def load_mtsamples():
    """Load and filter MTSamples knowledge base."""
    df = pd.read_csv("mtsamples.csv")
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    relevant = [
        "Cardiovascular / Pulmonary", "General Medicine",
        "Consult - History and Phy.", "Discharge Summary",
        "SOAP / Chart / Progress Notes", "Emergency Room Reports", "Nephrology"
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
            "specialty":   df_rag.iloc[idx]["medical_specialty"],
            "description": df_rag.iloc[idx]["description"],
            "text":        df_rag.iloc[idx]["transcription"][:600],
            "distance":    float(dist)
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

CHAIN-OF-THOUGHT: Before writing, reason through:
1. Which features are driving the risk score?
2. What do the retrieved notes say about similar cases?
3. What are the most important clinical actions?

FEW-SHOT FORMAT — follow exactly:
---
CLINICAL SUMMARY (For the Doctor):
Patient presents with [key features]. Risk of [X] indicates [level].
Primary drivers: [features]. Evidence suggests [finding]. Next steps: [actions].

PATIENT EXPLANATION (For the Patient):
Your heart check shows [simple finding]. Think of [analogy].
Main concerns: [simplified]. You can help by [actions].
---

CONSTRAINTS:
- CLINICAL: Medical terminology, specific values, precise.
- PATIENT: No jargon, everyday analogies, honest but reassuring.
- BOTH: Ground ONLY in retrieved notes. Do NOT fabricate.
- Keep each to 4-6 sentences maximum.

Generate both explanations now:
"""


def generate_explanation(prompt: str, api_key: str) -> str:
    """Call Gemini API to generate dual-audience explanation."""
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
    <div style='text-align:center; padding: 16px 0 24px 0;'>
        <div style='font-size:40px;'>🫀</div>
        <div style='font-size:18px; font-weight:700; color:#FFFFFF; margin-top:8px;'>CARA</div>
        <div style='font-size:11px; color:#48C9B0; letter-spacing:1px;'>CARDIOVASCULAR AI RISK ADVISOR</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Patient Clinical Features")
    st.markdown("<div style='font-size:11px; color:#7F9DB0; margin-bottom:16px;'>Adjust all values to match the patient profile</div>", unsafe_allow_html=True)

    # Feature inputs
    age      = st.slider("Age (years)", 20, 80, 55, 1)
    sex      = st.selectbox("Biological Sex", ["Male (1)", "Female (0)"])
    sex_val  = 1 if "Male" in sex else 0

    cp_options = {
        "Typical Angina (0)": 0,
        "Atypical Angina (1)": 1,
        "Non-anginal Pain (2)": 2,
        "Asymptomatic (3)": 3
    }
    cp = st.selectbox("Chest Pain Type", list(cp_options.keys()))
    cp_val = cp_options[cp]

    trestbps = st.slider("Resting Blood Pressure (mmHg)", 80, 220, 130, 1)
    chol     = st.slider("Serum Cholesterol (mg/dL)", 100, 600, 240, 1)

    fbs_opt = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No (0)", "Yes (1)"])
    fbs_val = 1 if "Yes" in fbs_opt else 0

    restecg_options = {"Normal (0)": 0, "ST-T Abnormality (1)": 1, "LV Hypertrophy (2)": 2}
    restecg = st.selectbox("Resting ECG", list(restecg_options.keys()))
    restecg_val = restecg_options[restecg]

    thalch  = st.slider("Max Heart Rate Achieved (bpm)", 60, 220, 150, 1)

    exang_opt = st.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"])
    exang_val = 1 if "Yes" in exang_opt else 0

    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, 0.1)

    slope_options = {"Upsloping (0)": 0, "Flat (1)": 1, "Downsloping (2)": 2}
    slope = st.selectbox("ST Slope", list(slope_options.keys()))
    slope_val = slope_options[slope]

    st.markdown("---")
    st.markdown("### Gemini API Key")
    api_key = st.text_input("Paste your API key", type="password", placeholder="AIza...")
    st.markdown("<div style='font-size:10px; color:#7F9DB0;'>Get free key at aistudio.google.com</div>", unsafe_allow_html=True)

    st.markdown("---")
    analyse_btn = st.button("🫀  Analyse Patient Risk", use_container_width=True)


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class='cara-header'>
    <div class='cara-badge'>CLINICAL DECISION SUPPORT</div>
    <h1 class='cara-title'>Cardiovascular AI Risk Advisor</h1>
    <p class='cara-subtitle'>
        <span class='status-dot'></span>
        Powered by Logistic Regression + RAG + Gemini 2.5 Flash-Lite
        &nbsp;|&nbsp; Heart Disease UCI Dataset &nbsp;|&nbsp; MTSamples Knowledge Base
    </p>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Loading CARA models..."):
    try:
        lr_model, scaler, feature_names, faiss_index, embed_model = load_models()
        df_rag = load_mtsamples()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure all model files are in the same folder as cara_app.py")
        models_loaded = False

if models_loaded:
    # Build feature vector in correct order
    feature_values = {
        "age": age, "sex": sex_val, "cp": cp_val,
        "trestbps": trestbps, "chol": chol, "fbs": fbs_val,
        "restecg": restecg_val, "thalch": thalch,
        "exang": exang_val, "oldpeak": oldpeak, "slope": slope_val
    }

    # Always show current patient features
    st.markdown("""
    <div class='section-header'>
        <h3>Current Patient Profile</h3>
        <p>Features entered via sidebar controls</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature tags
    tags_html = "".join([
        f"<span class='feature-tag'>{k}: {v}</span>"
        for k, v in feature_values.items()
    ])
    st.markdown(f"<div style='padding: 8px 0;'>{tags_html}</div>", unsafe_allow_html=True)

    # Quick risk preview (always visible)
    feature_vector = np.array([[feature_values[fn] for fn in feature_names]])
    feature_scaled = scaler.transform(feature_vector)
    risk_score     = lr_model.predict_proba(feature_scaled)[0][1]
    risk_color, risk_label, risk_class = get_risk_color(risk_score)

    col_risk, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])

    with col_risk:
        st.markdown(f"""
        <div class='risk-card {risk_class}'>
            <div style='font-size:12px; color:{risk_color}; font-weight:600; letter-spacing:2px; text-transform:uppercase;'>
                Risk Assessment
            </div>
            <div class='risk-score-number' style='color:{risk_color};'>
                {risk_score:.3f}
            </div>
            <div class='risk-label' style='color:{risk_color};'>{risk_label}</div>
            <div class='risk-description' style='color:#A8C5DA;'>
                Probability of cardiovascular disease
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{risk_score*100:.1f}%</div>
            <div class='metric-label'>Risk Probability</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        threshold_dist = abs(risk_score - 0.5)
        confidence = "HIGH" if threshold_dist > 0.25 else "MODERATE" if threshold_dist > 0.1 else "LOW"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:#F39C12;'>{confidence}</div>
            <div class='metric-label'>Confidence</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        category = "High Risk" if risk_score > 0.75 else "Low Risk" if risk_score < 0.25 else "Borderline"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='font-size:16px;'>{category}</div>
            <div class='metric-label'>Category</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Top risk drivers
    st.markdown("""
    <div class='section-header'>
        <h3>Top Risk Drivers</h3>
        <p>Features with highest odds ratios from Logistic Regression model</p>
    </div>
    """, unsafe_allow_html=True)

    odds_ratios = {
        "oldpeak (ST Depression)": 1.826,
        "exang (Exercise Angina)": 1.719,
        "sex (Biological Sex)":    1.682,
        "age":                     1.515,
        "fbs (Fasting Blood Sugar)":1.180,
        "trestbps (Blood Pressure)":1.053,
    }
    max_or = max(odds_ratios.values())
    cols_or = st.columns(2)
    for i, (feat, or_val) in enumerate(odds_ratios.items()):
        col = cols_or[i % 2]
        with col:
            bar_width = int((or_val / max_or) * 100)
            bar_color = "#E74C3C" if or_val > 1 else "#27AE60"
            st.markdown(f"""
            <div class='odds-bar-container'>
                <div class='odds-bar-label'>
                    <span style='color:#A8C5DA;'>{feat}</span>
                    <span style='color:{bar_color}; font-family:DM Mono; font-weight:600;'>OR {or_val}</span>
                </div>
                <div style='background:#061020; border-radius:4px; height:6px; overflow:hidden;'>
                    <div style='background:{bar_color}; width:{bar_width}%; height:100%; border-radius:4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # GENERATE EXPLANATION
    if analyse_btn:
        if not api_key:
            st.warning("Please enter your Gemini API key in the sidebar to generate explanations.")
        else:
            # Step 1: Build query and retrieve notes
            with st.spinner("Searching clinical knowledge base..."):
                query = build_clinical_query(feature_values)
                notes = retrieve_notes(query, faiss_index, embed_model, df_rag, k=3)

            # Show retrieved notes
            st.markdown("""
            <div class='section-header'>
                <h3>Retrieved Clinical Evidence</h3>
                <p>Top 3 most relevant notes from MTSamples knowledge base (1,576 cardiovascular notes)</p>
            </div>
            """, unsafe_allow_html=True)

            for i, note in enumerate(notes):
                st.markdown(f"""
                <div class='note-card'>
                    <span class='note-rank'>#{i+1} RETRIEVED</span>
                    <div class='note-specialty'>{note['specialty']}</div>
                    <div class='note-description'>{note['description']}</div>
                    <div class='note-text'>{note['text']}...</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Step 2: Generate explanation
            st.markdown("""
            <div class='section-header'>
                <h3>CARA Dual-Audience Explanation</h3>
                <p>Generated by Gemini 2.5 Flash-Lite — grounded in retrieved clinical evidence</p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Generating dual-audience explanation via Gemini..."):
                try:
                    prompt      = build_prompt(risk_score, risk_label, feature_values, notes)
                    explanation = generate_explanation(prompt, api_key)

                    # Split clinical and patient sections
                    clinical_text = ""
                    patient_text  = ""

                    if "CLINICAL SUMMARY" in explanation and "PATIENT EXPLANATION" in explanation:
                        parts = explanation.split("PATIENT EXPLANATION")
                        clinical_text = parts[0].replace("CLINICAL SUMMARY (For the Doctor):", "").replace("CLINICAL SUMMARY:", "").strip()
                        patient_text  = parts[1].replace("(For the Patient):", "").strip()
                        if patient_text.startswith(":"):
                            patient_text = patient_text[1:].strip()
                    else:
                        clinical_text = explanation
                        patient_text  = explanation

                    col_clin, col_pat = st.columns(2)

                    with col_clin:
                        st.markdown(f"""
                        <div class='explanation-clinical'>
                            <div class='explanation-label explanation-label-clinical'>
                                🩺 Clinical Summary — For the Doctor
                            </div>
                            <div class='explanation-text'>{clinical_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_pat:
                        st.markdown(f"""
                        <div class='explanation-patient'>
                            <div class='explanation-label explanation-label-patient'>
                                👤 Patient Explanation — For the Patient
                            </div>
                            <div class='explanation-text'>{patient_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Full explanation expander
                    with st.expander("View full Gemini response"):
                        st.text(explanation)

                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.info("Check your API key and make sure billing is set up at aistudio.google.com")

    else:
        # Prompt to click analyse
        st.markdown("""
        <div style='text-align:center; padding:40px; background:#0D1F3C; border:1px dashed #1B3A6B; border-radius:12px;'>
            <div style='font-size:32px; margin-bottom:12px;'>🫀</div>
            <div style='font-size:16px; color:#A8C5DA; font-weight:500;'>
                Adjust patient features in the sidebar, then click
            </div>
            <div style='font-size:20px; color:#48C9B0; font-weight:700; margin-top:8px;'>
                Analyse Patient Risk
            </div>
            <div style='font-size:12px; color:#7F9DB0; margin-top:8px;'>
                CARA will retrieve relevant clinical evidence and generate<br>
                a dual-audience explanation grounded in real clinical notes
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align:center; padding:16px; color:#3A5A7A; font-size:11px;'>
        CARA — Cardiovascular AI Risk Advisor &nbsp;|&nbsp;
        Generative AI Advanced Capstone &nbsp;|&nbsp;
        Raamalekshmi Murugan &nbsp;|&nbsp; April 2026
        <br><br>
        <span style='color:#2A4A6A;'>
        This system is a proof-of-concept for clinical decision support only.
        All outputs require physician review before any clinical action is taken.
        Not approved for autonomous diagnostic use.
        </span>
    </div>
    """, unsafe_allow_html=True)
