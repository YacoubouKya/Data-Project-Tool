# app.py
import streamlit as st
import pandas as pd
from modules import data_loader, eda, preprocessing, modeling, evaluation, reporting

# ------------------------
# ⚙️ Configuration de la page
# ------------------------
st.set_page_config(
    page_title="Data Project Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# 🎨 CSS personnalisé
# ------------------------
# ------------------------
# 🎨 CSS + Header personnalisé
# ------------------------
st.markdown("""
    <style>
    /* ===== HEADER FIXE ===== */
    .custom-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background-color: #1E3A5F; /* Bleu foncé */
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 40px;
        z-index: 9999;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.3);
    }

    .custom-header .logo {
        font-size: 22px;
        font-weight: bold;
        color: #FFD700; /* Jaune doré */
    }

    .custom-header .menu {
        display: flex;
        gap: 20px;
    }

    .custom-header .menu a {
        color: white;
        text-decoration: none;
        font-weight: 500;
        font-family: 'Segoe UI', sans-serif;
        transition: color 0.3s;
    }

    .custom-header .menu a:hover {
        color: #FFD700;
    }

    /* ===== Décalage contenu (éviter chevauchement) ===== */
    .block-container {
        padding-top: 80px !important;
    }

    /* ===== Fond global ===== */
    .stApp {
        background-color: #1E3A5F; /* Bleu foncé */
    }
    .block-container, .st-emotion-cache-18e3th9, .st-emotion-cache-1y4p8pa {
        background-color: transparent !important;
    }

    /* ===== Titres ===== */
    h1, h2, h3, h4 {
        color: #FFD700; /* Jaune doré */
        font-family: 'Segoe UI', sans-serif;
    }

    /* ===== Texte ===== */
    p, span, label, div {
        color: #FFFFFF !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ===== Sidebar ===== */
    [data-testid="stSidebar"] {
        background-color: #1569C7 !important;  
        color: yellow !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label {
        color: yellow !important;
    }

    /* ===== Boutons ===== */
    .stButton>button {
        background-color: #FFD700;  
        color: #1E3A5F;  
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FFA500;
        color: white;
    }

    /* ===== Upload ===== */
    [data-testid="stFileUploader"] {
        background-color: #FFD700 !important;
        border-radius: 10px;
        padding: 10px;
    }
    [data-testid="stFileUploader"] label {
        color: #1E3A5F !important;
        font-weight: bold;
    }

    /* ===== Radio & Selectbox ===== */
    div[role="radiogroup"] > label, .stSelectbox {
        background: #34495E !important;
        color: yellow !important;
        padding: 8px 15px;
        border-radius: 8px;
        margin: 3px 0;
        cursor: pointer;
    }
    div[role="radiogroup"] > label:hover {
        background: #1ABC9C !important;
    }
    </style>
""", unsafe_allow_html=True)

# Injection HTML du header
st.markdown("""
    <div class="custom-header">
        <div class="logo">🐍 Data Project Tool</div>
        <div class="menu">
            <a href="#">About</a>
            <a href="#">Documentation</a>
            <a href="#">Community</a>
            <a href="#">Success Stories</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# ------------------------
# 🎯 En-tête principal
# ------------------------
st.title("📊 Data Project Tool")
st.markdown("Bienvenue dans ton outil de projet data interactif 🚀")

# ------------------------
# 📌 Sidebar Navigation
# ------------------------
st.sidebar.title("📌 Navigation")
section = st.sidebar.radio(
    "Aller à :",
    ["📥 Chargement", "🔎 EDA", "🛠️ Prétraitement", "🤖 Modélisation", "📈 Évaluation", "📝 Reporting"]
)

# ------------------------
# Sections
# ------------------------
if section == "📥 Chargement":
    st.header("📥 Chargement des données")
    
    uploaded = st.file_uploader("Charger un fichier (CSV ou Excel)", type=["csv", "xlsx", "xls"])
    sep = ","
    sheet = None

    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            sep = st.selectbox("Séparateur CSV", options=[",", ";", "\t"], index=0)

        elif uploaded.name.lower().endswith((".xls", ".xlsx")):
            xls = pd.ExcelFile(uploaded)
            sheet = st.selectbox("Choisissez la feuille Excel", options=xls.sheet_names)

        df = data_loader.load_file(uploaded, sep=sep, sheet_name=sheet)
        if df is not None:
            st.session_state["data"] = df
            st.success("✅ Données chargées avec succès !")
            st.dataframe(df.head())

elif section == "🔎 EDA":
    st.header("🔎 Analyse exploratoire (EDA)")
    if "data" in st.session_state:
        eda.run_eda(st.session_state["data"])
    else:
        st.warning("⚠️ Chargez d'abord les données dans l'onglet Chargement.")

elif section == "🛠️ Prétraitement":
    st.header("🛠️ Prétraitement")
    if "data" in st.session_state:
        cleaned = preprocessing.run_preprocessing(st.session_state["data"])
        if cleaned is not None and not cleaned.equals(st.session_state["data"]):
            st.session_state["clean_data"] = cleaned
            st.success("✅ Données prétraitées avec succès !")
    else:
        st.warning("⚠️ Chargez d'abord les données.")

elif section == "🤖 Modélisation":
    st.header("🤖 Modélisation")
    if "data" in st.session_state:
        res = modeling.run_modeling(st.session_state["data"])
        if res:
            model, X_test, X_train, y_test, y_train = res
            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.success("✅ Modèle entraîné avec succès !")
    else:
        st.warning("⚠️ Chargez d'abord les données.")

elif section == "📈 Évaluation":
    st.header("📈 Évaluation du modèle")
    if "model" in st.session_state:
        evaluation.run_evaluation(
            st.session_state["model"], 
            st.session_state["X_test"], 
            st.session_state["y_test"]
        )
    else:
        st.warning("⚠️ Entraînez un modèle d'abord.")

elif section == "📝 Reporting":
    st.header("📝 Reporting")
    reporting.generate_report(st.session_state)
