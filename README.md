CARA — Cardiovascular AI Risk Advisor
An AI-Powered Clinical Decision Support System

What is CARA?
CARA is a Generative AI capstone project built as part of the TCS Generative AI Advanced course. It combines machine learning risk prediction with Retrieval Augmented Generation to produce dual-audience cardiovascular risk explanations — one for clinicians and one for patients — from a single system.
CARA was designed to address a critical gap in clinical AI: existing tools can predict cardiovascular risk but cannot explain it in a way that is meaningful to doctors or accessible to patients. CARA solves both problems simultaneously.

How It Works
CARA operates in two layers:
Layer 1 is machine learning. A Logistic Regression model trained on the Heart Disease UCI dataset (920 patients, 11 clinical features) predicts the probability of heart disease for a given patient. The model achieves a ROC AUC of 0.8987, outperforming an XGBoost comparison model (0.8839) while remaining fully interpretable through odds ratios.
Layer 2 is Retrieval Augmented Generation. The patient's clinical features are automatically converted into a natural language search query. FAISS searches through 1,576 cardiovascular-relevant MTSamples clinical notes using semantic similarity and retrieves the 3 most relevant ones. These notes, together with the risk score and patient features, are passed to Google Gemini 2.5 Flash-Lite which generates a dual-audience explanation grounded in the retrieved evidence.

System Architecture
Patient Features (11 clinical inputs)
        |
        v
Logistic Regression Model  -->  Risk Score + Risk Label
        |
        v
Feature-Grounded Query Builder
        |
        v
all-MiniLM-L6-v2 Embeddings  -->  384-dimensional query vector
        |
        v
FAISS IndexFlatL2  -->  Top-3 most relevant clinical notes
        |
        v
Gemini 2.5 Flash-Lite  -->  Dual-Audience Explanation
        |
        v
Clinical Summary (Doctor)  +  Patient Explanation (Patient)

Key Innovations
Dual-Audience Output: CARA generates two completely different explanations from a single API call. The doctor receives a clinical summary with specific values and odds ratios. The patient receives a clear explanation using relatable analogies grounded in retrieved clinical evidence.
Feature-Grounded Retrieval: Instead of searching based on a typed query, CARA automatically converts patient clinical features into a natural language search query. No patient identity is ever used — only clinical measurements.
RAG Prevents Hallucination: Gemini is explicitly constrained to only use the 3 retrieved clinical notes as evidence. This prevents fabrication of clinical findings. Average faithfulness score of 0.6875 across 20 diverse patients validates this approach.
Interpretability Over Black Box: Logistic Regression was chosen over XGBoost because odds ratios directly explain each feature's contribution to risk. A clinician can see that ST depression (oldpeak) has an odds ratio of 1.826, meaning every unit increase multiplies heart disease risk by 1.83 times.

Model Performance
MetricValueLogistic Regression ROC AUC0.8987XGBoost ROC AUC (comparison)0.8839RAG Faithfulness (avg, 20 patients)0.6875RAG Answer Relevancy (avg)0.9150Prediction Accuracy82% on 184 test patientsTraining Dataset920 patients, 11 featuresRAG Knowledge Base1,576 cardiovascular clinical notes

Datasets
Heart Disease UCI Dataset — Kaggle (originally from UCI Machine Learning Repository). 920 patients across 4 hospitals. 11 clinical features used after dropping high-missing columns (ca: 66% missing, thal: 53% missing).
MTSamples Medical Transcriptions — Kaggle (CC0 Public Domain License). 4,999 total notes filtered to 1,576 cardiovascular-relevant notes across 7 specialties including Cardiovascular/Pulmonary, General Medicine, Consult History, Discharge Summary, SOAP Notes, Emergency Room and Nephrology.

Technology Stack
ComponentTechnologyML Modelscikit-learn Logistic RegressionComparison ModelXGBoostEmbeddingsSentence Transformers all-MiniLM-L6-v2Vector SearchFAISS IndexFlatL2LLM GenerationGoogle Gemini 2.5 Flash-LiteDemo UIStreamlitDevelopmentPython 3.10, Google Colab

Setup and Installation
Step 1 — Install required libraries:
pip install streamlit joblib faiss-cpu sentence-transformers google-generativeai pandas numpy
Step 2 — Place the following files in the same folder as cara_app.py:
lr_model.joblib
scaler.joblib
feature_names.joblib
faiss.index
mtsamples.csv
Step 3 — Get a free Gemini API key from https://aistudio.google.com
Step 4 — Run the app:
streamlit run cara_app.py
Step 5 — Enter your Gemini API key in the sidebar, adjust patient features using the sliders and dropdowns, and click Analyse Patient.

Responsible AI
CARA is a proof of concept and a decision support tool only. It is not approved for clinical use. Every output requires physician review before any clinical action is taken.
NIST AI RMF alignment is documented throughout the system. Gender bias is acknowledged — the training dataset is 79% male and may underestimate risk for female patients presenting with atypical symptoms. No patient identity is used at any stage of the pipeline.

Author
Raamalekshmi Murugan
Generative AI Advanced — Capstone Project
