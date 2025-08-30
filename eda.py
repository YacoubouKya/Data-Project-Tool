# eda.py
# modules/eda.py
import streamlit as st
import pandas as pd

def run_eda(df: pd.DataFrame):
    """
    Interface EDA :
    - Aperçu
    - Profiling (ydata_profiling via streamlit_pandas_profiling si dispo)
    - Visualisations basiques (subset)
    """
    st.subheader("Aperçu général")
    st.write("Dimensions :", df.shape)
    st.dataframe(df.head())

    # Quick stats
    st.markdown("**Statistiques descriptives (numériques)**")
    st.dataframe(df.describe().T)

    # Try to show profiling inline
    if st.button("Générer Profiling Report (interactive)"):
        try:
            from ydata_profiling import ProfileReport
            from streamlit_pandas_profiling import st_profile_report
            # allow user to pick minimal or full
            #minimal = st.checkbox("Mode minimal (plus rapide)", value=True)
            prof = ProfileReport(df, title="Profiling EDA", explorative=True)
            st_profile_report(prof)
        except Exception as e:
            st.warning("pandas_profiling ou streamlit_pandas_profiling non installé ou erreur.")
            st.write("Erreur :", e)
            st.info("Vous pouvez installer: pip install ydata-profiling streamlit-pandas-profiling")

    # Basic visual plots
    if st.button("Générer visualisations basiques (hist / corr)"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        num_cols = df.select_dtypes(include="number").columns.tolist()
        n = min(6, len(num_cols))
        if n == 0:
            st.info("Aucune variable numérique détectée pour tracer des histogrammes.")
        else:
            cols_sample = num_cols[:n]
            st.write("Histogrammes (quelques variables numériques)")
            fig, axs = plt.subplots(n, 1, figsize=(6, 3*n))
            if n == 1:
                axs = [axs]
            for ax, c in zip(axs, cols_sample):
                sns.histplot(df[c].dropna(), kde=False, ax=ax)
                ax.set_title(c)
            st.pyplot(fig)

        if len(num_cols) >= 2:
            st.write("Heatmap de corrélation (subset)")
            corr = df[num_cols[:min(8, len(num_cols))]].corr()
            fig2, ax2 = plt.subplots(figsize=(6,5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
            st.pyplot(fig2)
