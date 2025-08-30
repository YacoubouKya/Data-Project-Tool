# reporting.py
# modules/reporting.py
import streamlit as st
import json
import os
from typing import Dict

OUT_DIR = "outputs/reports"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_report(session_state: Dict):
    """
    Génère un rapport HTML simple consolidé à partir du session_state Streamlit.
    On inclut : aperçu, info preprocessing (si présent), point modèle (si présent).
    """
    st.subheader("Générer rapport consolidé")
    title = st.text_input("Titre du rapport", value="Rapport consolidé")
    if st.button("Créer rapport HTML"):
        html = ["<html><head><meta charset='utf-8'><title>"+title+"</title></head><body>"]
        html.append(f"<h1>{title}</h1>")
        # Data overview
        if "data" in session_state:
            df = session_state["data"]
            html.append(f"<h2>Données</h2><p>Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes</p>")
            html.append(df.head(5).to_html(index=False))

        # Cleaned
        if "clean_data" in session_state:
            cdf = session_state["clean_data"]
            html.append(f"<h2>Données préparées</h2><p>Dimensions : {cdf.shape[0]} × {cdf.shape[1]}</p>")
            html.append(cdf.head(5).to_html(index=False))

        # Model info
        if "model" in session_state:
            html.append("<h2>Modèle entraîné</h2><p>Modèle sauvegardé en outputs/models/ (voir console)</p>")

        out_path = os.path.join(OUT_DIR, "report_consolidated.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        st.success(f"Rapport généré : {out_path}")
        st.markdown(f"[Ouvrir le rapport]({out_path})")
