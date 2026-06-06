import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# PAGE CONFIG
# ==================================================
st.markdown("""
<style>
    /* =========================
       MAIN CONTAINER
    ========================= */
    .block-container {
        max-width: 95% !important;
        padding-top: 6rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* =========================
       PAGE BACKGROUND
    ========================= */
    .stApp {
        background-color: #F8FAFC;
    }

    /* =========================
       HERO CARD
    ========================= */
    .hero-card {
        background: linear-gradient(
            135deg,
            #0F172A 0%,
            #1E293B 100%
        );
        width: 100%;
        padding: 35px;
        border-radius: 20px;
        margin-bottom: 25px;
        border: 1px solid #334155;
        box-shadow:
            0 10px 25px rgba(15,23,42,0.15);
        color: white;
    }
    .hero-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        color: #CBD5E1 !important;
        font-size: 18px;
    }

    /* =========================
       METRIC CARD
    ========================= */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        padding: 18px;
        box-shadow:
            0 2px 8px rgba(15,23,42,0.05);
    }
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600 !important;
    }

    /* =========================
       FORM CARD
    ========================= */
    [data-testid="stForm"] {
        width: 100%;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 16px;
        padding: 30px;
        box-shadow:
            0 4px 12px rgba(15,23,42,0.04);
    }

    /* =========================
       INPUT BOX
    ========================= */
    .stNumberInput input {
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="select"] > div {
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
    }

    /* =========================
       BUTTON
    ========================= */
    .stButton button,
    .stFormSubmitButton button {
        width: 100% !important;
        min-height: 60px !important;
        background: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease;
    }

    .stButton button *,
    .stFormSubmitButton button * {
        color: #FFFFFF !important;
    }
    .stButton button:hover,
    .stFormSubmitButton button:hover {
        background: #334155 !important;
        border: 1px solid #475569 !important;
        box-shadow:
            0 4px 12px rgba(15,23,42,0.20);
    }

    /* =========================
       HEADINGS
    ========================= */
    h1, h2, h3, h4 {
        color: #0F172A !important;
    }
    p, label {
        color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD ARTEFAK MODEL (Pastikan fungsi ini ada)
# ==================================================
@st.cache_resource
def load_all_artifacts():
    try:
        model = joblib.load("model.pkl")
        preprocessor = joblib.load("preprocessor.pkl")
        selector = joblib.load("selector.pkl")
        meta = joblib.load("meta.pkl")
        with open("threshold.txt", "r") as f:
            threshold = float(f.read().strip())
        return model, preprocessor, selector, meta, threshold
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        st.stop()

model, preprocessor, selector, meta, threshold = load_all_artifacts()

# ==================================================
# MAPPING DATA
# ==================================================
AGE_MAP = {1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+"}
EDU_MAP = {1: "Tidak Sekolah", 2: "SD", 3: "SMP", 4: "SMA", 5: "Diploma/Sarjana", 6: "Pascasarjana"}
INC_MAP = {1: "< $10k", 2: "$10k-$15k", 3: "$15k-$20k", 4: "$20k-$25k", 5: "$25k-$35k", 6: "$35k-$50k", 7: "$50k-$75k", 8: "> $75k"}
DIAB_MAP = {0: "Bukan Diabetes", 1: "Pre-Diabetes", 2: "Diabetes"}
GEN_MAP = {1: "Sangat Baik", 2: "Baik", 3: "Sedang", 4: "Buruk", 5: "Sangat Buruk"}

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div class="hero-card">
    <div class="hero-title">❤️ Heart Disease Risk Advisor</div>
    <div class="hero-subtitle">Sistem deteksi dini risiko penyakit jantung berbasis Machine Learning.</div>
</div>
""", unsafe_allow_html=True)

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Model Type", "Random Forest")
col2.metric("Decision Threshold", f"{threshold:.2f}")
col3.metric("Prediction Target", "Heart Disease")

st.markdown("<br>", unsafe_allow_html=True)

# ==================================================
# FORM INPUT
# ==================================================
with st.form("heart_health_form"):
    st.subheader("📝 Lengkapi Data Kesehatan")
    c1, c2 = st.columns(2)
    with c1:
        age = st.selectbox("Rentang Usia", options=list(AGE_MAP.keys()), format_func=lambda x: AGE_MAP[x])
        sex = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Wanita" if x == 0 else "Pria")
        education = st.selectbox("Pendidikan Terakhir", options=list(EDU_MAP.keys()), format_func=lambda x: EDU_MAP[x])
        income = st.selectbox("Pendapatan Tahunan", options=list(INC_MAP.keys()), format_func=lambda x: INC_MAP[x])
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=99.0, value=24.5, step=0.1)
    
    with c2:
        high_bp = st.selectbox("Riwayat Darah Tinggi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        high_chol = st.selectbox("Riwayat Kolesterol Tinggi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")
        diabetes = st.selectbox("Riwayat Diabetes", options=list(DIAB_MAP.keys()), format_func=lambda x: DIAB_MAP[x])
        gen_hlth = st.selectbox("Kondisi Kesehatan Umum", options=list(GEN_MAP.keys()), format_func=lambda x: GEN_MAP[x])
        smoker = st.selectbox("Perokok", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

    submit = st.form_submit_button("🔍 Analisis Risiko Sekarang", use_container_width=True)

# ==================================================
# PREDICTION ENGINE
# ==================================================
if submit:
    # Logika yang sama seperti sebelumnya...
    input_data = pd.DataFrame([{
        'HighBP': high_bp, 'HighChol': high_chol, 'CholCheck': 1, 'BMI': bmi,
        'Smoker': smoker, 'Stroke': 0, 'Diabetes': diabetes, 'PhysActivity': 1,
        'Fruits': 1, 'Veggies': 1, 'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1, 'NoDocbcCost': 0, 'GenHlth': gen_hlth,
        'MentHlth': 0, 'PhysHlth': 0, 'DiffWalk': 0,
        'Sex': sex, 'Age': age, 'Education': education, 'Income': income
    }])

    # Preprocessing
    if 'batas_outlier' in meta:
        for col, (l, u) in meta['batas_outlier'].items():
            input_data[col] = input_data[col].clip(l, u)
    if 'kolom_log' in meta:
        for col in meta['kolom_log']:
            input_data[f"{col}_log"] = np.log1p(input_data[col])

    # Prediction
    try:
        X_trans = preprocessor.transform(input_data)
        X_sel = selector.transform(X_trans)
        prob = model.predict_proba(X_sel)[0, 1]
        is_risk = prob >= threshold

        st.divider()
        st.subheader("📋 Hasil Analisis")
        st.progress(float(prob))
        
        c1, c2 = st.columns(2)
        c1.metric("Probabilitas", f"{prob*100:.2f}%")
        c2.metric("Kategori", "TINGGI" if is_risk else "RENDAH")
        
        if is_risk:
            st.error("⚠️ **Risiko tinggi terdeteksi.** Mohon segera konsultasikan dengan tenaga medis.")
        else:
            st.success("✅ **Risiko rendah.** Tetap pertahankan pola hidup sehat.")
    except Exception as e:
        st.error(f"Error: {e}")