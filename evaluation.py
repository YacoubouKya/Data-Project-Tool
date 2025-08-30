# evaluation.py
# modules/evaluation.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from modules.utils.metrics import classification_metrics, regression_metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc

def run_evaluation(model, X_test, y_test):
    st.subheader("Évaluation du modèle")
    preds = model.predict(X_test)
    # detect task type
    if y_test.dtype == "object" or y_test.nunique() < 20:
        st.write("Classification — métriques :")
        metrics = classification_metrics(y_test, preds)
        st.json(metrics)
        cm = confusion_matrix(y_test, preds)
        st.write("Matrice de confusion")
        st.write(cm)
    else:
        st.write("Régression — métriques :")
        metrics = regression_metrics(y_test, preds)
        st.json(metrics)
        # plot preds vs true
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(y_test, preds, alpha=0.6)
        ax.set_xlabel("Vrai")
        ax.set_ylabel("Prédit")
        ax.set_title("Prédit vs Réel")
        st.pyplot(fig)
