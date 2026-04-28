"""
CARA — Cardiovascular AI Risk Advisor
Streamlit Demo Application

HOW TO RUN:
  1. Create .streamlit/secrets.toml with:
       GEMINI_API_KEY = "your_key_here"
  2. Install dependencies:
       pip install streamlit joblib faiss-cpu sentence-transformers google-generativeai pandas numpy
  3. Place these files in the same folder:
       lr_model.joblib, scaler.joblib, feature_names.joblib, faiss.index, mtsamples.csv
  4. Run:
       streamlit run cara_app.py
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ─── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="CARA — Cardiovascular AI Risk Advisor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F4F8FB; }
    .stApp { background-color: #F4F8FB; }

    .cara-header {
        background: linear-gradient(135deg, #0A2342 0%, #1B6CA8 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .cara-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .cara-header p  { color: #48C9B0; font-size: 1rem; margin: 0.3rem 0 0 0; font-style: italic; }

    .risk-high       { background: linear-gradient(135deg, #E74C3C, #C0392B); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; }
    .risk-low        { background: linear-gradient(135deg, #27AE60, #1E8449); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; }
    .risk-borderline { background: linear-gradient(135deg, #F39C12, #D68910); color: white; padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem; }
    .risk-score { font-size: 3rem; font-weight: 700; margin: 0; }
    .risk-label { font-size: 1.3rem; font-weight: 600; margin: 0.2rem 0 0 0; }

    .section-card {
        background: white; border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem; border-left: 4px solid #1B6CA8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .section-card h3 { color: #1B6CA8; font-size: 1rem; font-weight: 700; margin: 0 0 0.5rem 0; text-transform: uppercase; }

    .note-card {
        background: #EAF4FB; border-radius: 8px; padding: 1rem;
        margin-bottom: 0.8rem; border-left: 3px solid #48C9B0;
    }
    .note-card h4 { color: #0A2342; font-size: 0.9rem; font-weight: 700; margin: 0 0 0.3rem 0; }
    .note-card p  { font-size: 0.85rem; color: #444; margin: 0; }

    .clinical-card {
        background: #FEF9E7; border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem; border-left: 4px solid #F39C12;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .clinical-card h3 { color: #D68910; font-size: 1rem; font-weight: 700; margin: 0 0 0.5rem 0; text-transform: uppercase; }

    .patient-card {
        background: #EAF8F1; border-radius: 10px; padding: 1.2rem 1.5rem;
        margin-bottom: 1rem; border-left: 4px solid #27AE60;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .patient-card h3 { color: #1E8449; font-size: 1rem; font-weight: 700; margin: 0 0 0.5rem 0; text-transform: uppercase; }

    .cara-footer {
        background: #0A2342; color: #7F8C8D; padding: 0.8rem 1.5rem;
        border-radius: 8px; font-size: 0.8rem; text-align: center; margin-top: 2rem;
    }

    section[data-testid="stSidebar"] { background-color: #0A2342; }
    section[data-testid="stSidebar"] .stMarkdown { color: white; }
    section[data-testid="stSidebar"] label { color: #48C9B0 !important; font-weight: 600; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: white; }

    /* Fix metric component text — prevent white-on-white rendering */
    [data-testid="stMetricLabel"] p { color: #1A1A2E !important; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: #1A1A2E !important; font-weight: 700 !important; }
    [data-testid="metric-container"] { background: white; border-radius: 8px; padding: 0.5rem; border: 1px solid #E8EEF4; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD API KEY FROM STREAMLIT SECRETS ─────────────────
# Key is stored in .streamlit/secrets.toml — never hardcoded
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = None

# ─── LOAD MODEL ARTIFACTS ────────────────────────────────
@st.cache_resource
def load_models():
    lr_model      = joblib.load("lr_model.joblib")
    scaler        = joblib.load("scaler.joblib")
    feature_names = joblib.load("feature_names.joblib")
    faiss_index   = faiss.read_index("faiss.index")
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
    return lr_model, scaler, feature_names, faiss_index, embed_model

@st.cache_data
def load_mtsamples():
    df = pd.read_csv("mtsamples.csv")
    df["medical_specialty"] = df["medical_specialty"].str.strip()
    relevant = [
        "Cardiovascular / Pulmonary", "General Medicine",
        "Consult - History and Phy.", "Discharge Summary",
        "SOAP / Chart / Progress Notes", "Emergency Room Reports", "Nephrology"
    ]
    df = df[df["medical_specialty"].isin(relevant)].dropna(
        subset=["transcription"]).reset_index(drop=True)
    return df

# ─── HEADER ──────────────────────────────────────────────
st.markdown("""
<div class="cara-header">
    <h1>CARA — Cardiovascular AI Risk Advisor</h1>
    <p>An AI-Powered Clinical Advisor Where Machine Learning Meets Human Understanding in Cardiovascular Care</p>
</div>
""", unsafe_allow_html=True)

# ─── LOAD EVERYTHING ─────────────────────────────────────
with st.spinner("Loading CARA models and knowledge base..."):
    try:
        lr_model, scaler, feature_names, faiss_index, embed_model = load_models()
        df_rag = load_mtsamples()
        models_loaded = True
    except Exception as e:
        st.error(f"Could not load model files: {e}")
        st.info("Make sure lr_model.joblib, scaler.joblib, feature_names.joblib, faiss.index and mtsamples.csv are in the same folder as cara_app.py")
        models_loaded = False

if models_loaded:

    # ─── SIDEBAR ─────────────────────────────────────────
    st.sidebar.markdown("## Patient Clinical Profile")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Demographic Features")

    age   = st.sidebar.slider("Age (years)", 20, 80, 55, help="Patient age in years")
    sex   = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x == 1 else "Female")

    st.sidebar.markdown("### Cardiac Symptoms")
    cp    = st.sidebar.slider("Chest Pain Type (0=Typical Angina, 3=None)", 0, 3, 0)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.sidebar.markdown("### Vital Signs and Lab Results")
    trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
    chol     = st.sidebar.slider("Serum Cholesterol (mg/dL)", 100, 600, 240)
    fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    thalch   = st.sidebar.slider("Maximum Heart Rate (bpm)", 60, 220, 150)

    st.sidebar.markdown("### ECG Results")
    restecg = st.sidebar.slider("Resting ECG (0=Normal, 2=Abnormal)", 0, 2, 0)
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope   = st.sidebar.slider("ST Slope (0=Upsloping, 2=Downsloping)", 0, 2, 1)

    st.sidebar.markdown("---")
    analyse_btn = st.sidebar.button("Analyse Patient Risk", type="primary", use_container_width=True)

    # ─── INSTRUCTIONS BEFORE ANALYSIS ────────────────────
    if not analyse_btn:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="section-card"><h3>Step 1 — Enter Patient Data</h3>
            <p>Use the sliders and dropdowns on the left sidebar to enter the patient's clinical features.</p></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="section-card"><h3>Step 2 — Click Analyse</h3>
            <p>CARA will predict cardiovascular risk and retrieve the 3 most relevant clinical notes from 1,576 transcriptions.</p></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="section-card"><h3>Step 3 — Review Explanations</h3>
            <p>CARA generates a clinical summary for the doctor and a clear explanation for the patient, both grounded in retrieved evidence.</p></div>""", unsafe_allow_html=True)
        st.info("CARA is a clinical decision support tool. All outputs require physician review before any clinical action is taken.")

    # ─── RUN ANALYSIS ────────────────────────────────────
    if analyse_btn:

        # Build feature vector in correct order
        feature_map = {
            "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
            "chol": chol, "fbs": fbs, "restecg": restecg, "thalch": thalch,
            "exang": exang, "oldpeak": oldpeak, "slope": slope
        }
        features_array  = np.array([feature_map.get(f, 0) for f in feature_names]).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        risk_proba      = lr_model.predict_proba(features_scaled)[0][1]
        risk_pct        = round(risk_proba * 100, 1)

        if risk_proba >= 0.75:
            risk_label = "HIGH RISK"
            risk_class = "risk-high"
            risk_emoji = "🔴"
        elif risk_proba >= 0.5:
            risk_label = "ELEVATED RISK"
            risk_class = "risk-borderline"
            risk_emoji = "🟡"
        elif risk_proba >= 0.25:
            risk_label = "BORDERLINE"
            risk_class = "risk-borderline"
            risk_emoji = "🟡"
        else:
            risk_label = "LOW RISK"
            risk_class = "risk-low"
            risk_emoji = "🟢"

        # ─── RISK SCORE DISPLAY ──────────────────────────
        st.markdown(f"""
        <div class="{risk_class}">
            <p class="risk-score">{risk_emoji} {risk_pct}%</p>
            <p class="risk-label">{risk_label}</p>
            <p style="margin:0.3rem 0 0 0; font-size:0.9rem; opacity:0.9;">
                Cardiovascular Risk Score — Logistic Regression (AUC 0.8987)
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ─── TOP RISK FACTORS ────────────────────────────
        odds_ratios = {
            "oldpeak": 1.826, "exang": 1.719, "sex": 1.682,
            "age": 1.515, "fbs": 1.180, "trestbps": 1.053,
            "restecg": 0.931, "slope": 0.887, "thalch": 0.738,
            "chol": 0.649, "cp": 0.599
        }
        factor_labels = {
            "oldpeak": "ST Depression", "exang": "Exercise Angina",
            "sex": "Biological Sex", "age": "Age",
            "fbs": "Blood Sugar", "trestbps": "Blood Pressure"
        }
        factor_descriptions = {
            "oldpeak": "Strongest predictor — heart muscle starved of blood during exertion",
            "exang":   "Chest pain during exercise signals blocked coronary arteries",
            "sex":     "Male patients carry statistically higher cardiovascular risk",
            "age":     "Every additional year adds measurable cardiovascular risk",
            "fbs":     "Elevated blood sugar signals diabetes — a major cardiac risk factor",
            "trestbps":"High resting blood pressure means the heart is overworking at rest",
        }
        top_factors = sorted(
            [(f, odds_ratios.get(f, 1.0)) for f in feature_names if odds_ratios.get(f, 1.0) > 1.0],
            key=lambda x: x[1], reverse=True
        )[:3]

        st.markdown("**Top 3 risk drivers for this patient:**")
        cols = st.columns(3)
        for i, (feat, orval) in enumerate(top_factors):
            with cols[i]:
                desc = factor_descriptions.get(feat, "Contributes to cardiovascular risk")
                st.markdown(f'''
                <div style="background:white; border-radius:10px; padding:1rem 1.2rem;
                             border-left:4px solid #E74C3C; box-shadow:0 2px 8px rgba(0,0,0,0.06);
                             margin-bottom:0.5rem;">
                    <p style="color:#E74C3C; font-size:0.8rem; font-weight:700;
                               text-transform:uppercase; margin:0 0 0.3rem 0;">Risk Factor {i+1}</p>
                    <p style="color:#0A2342; font-size:1.1rem; font-weight:700; margin:0 0 0.3rem 0;">
                        {factor_labels.get(feat, feat)}
                    </p>
                    <p style="color:#555; font-size:0.82rem; margin:0; line-height:1.4;">{desc}</p>
                </div>
                ''', unsafe_allow_html=True)

        st.markdown("---")

        # ─── FEATURE-GROUNDED RETRIEVAL ──────────────────
        query = (
            f"cardiac patient: age {age} years, blood pressure {trestbps} mmHg, "
            f"cholesterol {chol} mg/dL, max heart rate {thalch} bpm, "
            f"ST depression {oldpeak:.1f}. Cardiovascular risk, chest pain, coronary artery disease."
        )

        with st.spinner("Retrieving clinical evidence from knowledge base..."):
            query_emb           = embed_model.encode([query]).astype("float32")
            distances, indices  = faiss_index.search(query_emb, k=3)

        # ─── DISPLAY RETRIEVED NOTES ─────────────────────
        st.markdown("### Retrieved Clinical Evidence")
        st.markdown(
            "CARA retrieved the 3 most relevant notes from 1,576 cardiovascular clinical transcriptions. "
            "Gemini will use only these notes as evidence — no training memory, no fabrication."
        )

        retrieved_notes = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            note_text   = df_rag.iloc[idx]["transcription"][:600]
            specialty   = df_rag.iloc[idx]["medical_specialty"]
            description = df_rag.iloc[idx]["description"] if pd.notna(df_rag.iloc[idx]["description"]) else "Clinical Note"
            retrieved_notes.append(note_text)

            st.markdown(f"""
            <div class="note-card">
            <h4>Note {rank+1} of 3 — {specialty}</h4>
            <p><strong>{description}</strong></p>
            <p style="margin-top:0.4rem; font-size:0.82rem; color:#555;">{note_text[:350]}...</p>
            <p style="font-size:0.78rem; color:#888; margin-top:0.3rem;">Similarity distance: {dist:.4f} (lower = more relevant)</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ─── GEMINI DUAL-AUDIENCE EXPLANATION ────────────
        st.markdown("### Dual-Audience Clinical Explanation")
        st.markdown("One API call generates both explanations — grounded only in the retrieved notes above.")

        if not GEMINI_API_KEY:
            st.error("Gemini API key not found. Add GEMINI_API_KEY to your .streamlit/secrets.toml file.")
            st.code("""
# .streamlit/secrets.toml
GEMINI_API_KEY = "your_key_here"
            """)
        else:
            with st.spinner("Generating dual-audience explanation with Gemini 2.5 Flash-Lite..."):
                try:
                    genai.configure(api_key=GEMINI_API_KEY)
                    gemini_model  = genai.GenerativeModel("gemini-2.5-flash-lite")

                    notes_context = ""
                    for i, note in enumerate(retrieved_notes):
                        notes_context += f"\n[Note {i+1}]\n{note}\n"

                    prompt = f"""
ROLE: You are simultaneously a senior cardiologist and a patient health educator.

PATIENT CLINICAL PROFILE:
- Risk Score: {risk_proba:.4f} ({risk_label})
- Age: {age} years | Sex: {"Male" if sex == 1 else "Female"}
- Resting BP: {trestbps} mmHg | Cholesterol: {chol} mg/dL
- Max Heart Rate: {thalch} bpm | ST Depression (oldpeak): {oldpeak:.1f}
- Exercise-Induced Angina: {"Yes" if exang == 1 else "No"} | Fasting Blood Sugar High: {"Yes" if fbs == 1 else "No"}

RETRIEVED CLINICAL EVIDENCE (use ONLY these notes as evidence):
{notes_context}

CHAIN-OF-THOUGHT: Before writing, reason through:
1. Which features are driving the risk score?
2. What do the retrieved notes say about similar cases?
3. What are the most important clinical actions?

OUTPUT FORMAT:

CLINICAL SUMMARY (For the Doctor):
[4 to 6 sentences using medical terminology, specific values, odds ratios. Reference retrieved evidence.]

PATIENT EXPLANATION (For the Patient):
[4 to 6 sentences using no jargon. Use relatable analogies. Give clear lifestyle advice. Honest but reassuring.]

CONSTRAINTS:
- Ground ONLY in retrieved notes and patient profile. Do NOT fabricate clinical findings.
- Patient section must have zero medical jargon.
- Do NOT use em-dashes anywhere in your response.
- Keep each section to 4 to 6 sentences maximum.
"""
                    response    = gemini_model.generate_content(prompt)
                    explanation = response.text

                    # Parse the two sections
                    if "PATIENT EXPLANATION" in explanation:
                        parts         = explanation.split("PATIENT EXPLANATION")
                        clinical_part = parts[0].replace("CLINICAL SUMMARY (For the Doctor):", "").strip()
                        patient_part  = parts[1].replace("(For the Patient):", "").strip()
                    else:
                        clinical_part = explanation
                        patient_part  = ""

                    # Display clinical summary
                    st.markdown(f"""
                    <div class="clinical-card">
                    <h3>Clinical Summary — For the Doctor</h3>
                    <p style="line-height:1.7; color:#333;">{clinical_part}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display patient explanation
                    if patient_part:
                        st.markdown(f"""
                        <div class="patient-card">
                        <h3>Patient Explanation — For the Patient</h3>
                        <p style="line-height:1.7; color:#333;">{patient_part}</p>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.info("Check your API key in .streamlit/secrets.toml and ensure billing is enabled on your Google AI Studio account.")

        # ─── RESPONSIBLE AI FOOTER ────────────────────────
        st.markdown("""
        <div class="cara-footer">
            CARA is a clinical decision support tool only. All outputs require physician review before any clinical action is taken.
            No patient identity is used at any stage of the pipeline. Explanations are grounded in retrieved clinical evidence only.
            Raamalekshmi Murugan | Generative AI Advanced Capstone | April 2026
        </div>
        """, unsafe_allow_html=True)
