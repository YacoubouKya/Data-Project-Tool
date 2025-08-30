# preprocessing.py

# modules/preprocessing.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler

SCALERS = {
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(),
    "robust": RobustScaler(),
    "none": None
}

def run_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interface interactive de prétraitement.
    Retourne un DataFrame "nettoyé" (mais sans encodage one-hot expansé — on garde types).
    Pour la modélisation on renverra la pipeline (ColumnTransformer) séparément via run_preprocessing_pipeline.
    """
    st.subheader("Prétraitement interactif")
    st.write("Dimensions initiales:", df.shape)

    # Options simples
    drop_duplicates = st.checkbox("Supprimer les doublons", value=True)
    drop_thresh = st.slider("Supprimer colonnes avec > X% missing (0=none)", 0, 100, 100) / 100.0
    numeric_strategy = st.selectbox("Imputation numérique", ["median", "mean", "constant"])
    categorical_strategy = st.selectbox("Imputation catégorielle", ["most_frequent", "constant"])
    parse_dates = st.checkbox("Parser automatiquement les colonnes date (tentative)", value=True)
    cap_outliers = st.checkbox("Appliquer capping IQR sur numériques", value=False)

    if st.button("Appliquer prétraitement"):
        df2 = df.copy()

        # parse dates heuristique
        if parse_dates:
            for c in df2.select_dtypes(include="object").columns:
                try:
                    parsed = pd.to_datetime(df2[c], errors="coerce", infer_datetime_format=True)
                    if parsed.notna().mean() > 0.8:
                        df2[c] = parsed
                except Exception:
                    pass

        if drop_duplicates:
            before = len(df2)
            df2 = df2.drop_duplicates()
            st.write(f"Doublons supprimés : {before - len(df2)}")

        # drop columns with too many missing
        if drop_thresh < 1.0:
            keep = [c for c in df2.columns if df2[c].isna().mean() <= drop_thresh]
            dropped = set(df2.columns) - set(keep)
            df2 = df2[keep]
            st.write(f"Colonnes supprimées pour missing > {drop_thresh*100:.0f}% :", list(dropped))

        # simple imputations
        num_cols = df2.select_dtypes(include="number").columns.tolist()
        cat_cols = df2.select_dtypes(include=["object", "category"]).columns.tolist()

        if num_cols:
            if numeric_strategy == "median":
                df2[num_cols] = df2[num_cols].fillna(df2[num_cols].median())
            elif numeric_strategy == "mean":
                df2[num_cols] = df2[num_cols].fillna(df2[num_cols].mean())
            else:
                df2[num_cols] = df2[num_cols].fillna(0)

        if cat_cols:
            if categorical_strategy == "most_frequent":
                for c in cat_cols:
                    df2[c] = df2[c].fillna(df2[c].mode().iloc[0] if not df2[c].mode().empty else "Missing")
            else:
                df2[cat_cols] = df2[cat_cols].fillna("Missing")

        # capping outliers IQR
        if cap_outliers and num_cols:
            for c in num_cols:
                q1 = df2[c].quantile(0.25)
                q3 = df2[c].quantile(0.75)
                iqr = q3 - q1
                low = q1 - 1.5 * iqr
                high = q3 + 1.5 * iqr
                df2[c] = df2[c].clip(lower=low, upper=high)

        st.success("Prétraitement appliqué.")
        st.write("Dimensions finales:", df2.shape)
        st.dataframe(df2.head())
        return df2

    st.info("Configurer les options et cliquer sur 'Appliquer prétraitement'.")
    return df  # si non appliqué on renvoie original
