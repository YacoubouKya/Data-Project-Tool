# modeling.py
# modules/modeling.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
from typing import Tuple, Any
from modules.preprocessing import SCALERS
from modules.utils import helpers
from modules.utils.metrics import classification_metrics, regression_metrics

def run_modeling(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    UI for modeling:
    - choose target column
    - choose task type or autodetect
    - choose simple model and hyperparam quick-scan
    Returns trained estimator and train/test splits (X_test,y_test usable by evaluation).
    """
    st.subheader("Modélisation interactive")

    cols = df.columns.tolist()
    target = st.selectbox("Choisir la variable cible", [""] + cols)
    if not target:
        st.info("Sélectionne une variable cible pour lancer l'entraînement.")
        st.stop()

    X = df.drop(columns=[target])
    y = df[target]

    # detect task
    task = st.selectbox("Type de tâche", ["auto", "classification", "regression"], index=0)
    if task == "auto":
        if y.dtype == "O" or (y.nunique() <= 20 and y.nunique() / len(y) < 0.1):
            task = "classification"
        else:
            task = "regression"
    st.write("Tâche choisie :", task)

    test_size = st.slider("Taille test (%)", 5, 50, 20) / 100.0
    random_state = st.number_input("Seed aléatoire", value=42)

    model_choice = st.selectbox("Modèle simple", ["auto", "random_forest", "linear/logistic"])
    do_scale = st.checkbox("Standardiser les numériques (StandardScaler)", value=True)

    if st.button("Lancer entraînement"):
        # build simple preprocessor using ColumnTransformer (numerics scaled if asked)
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        X[cat_cols] = X[cat_cols].astype(str)

        num_steps = []
        if do_scale and num_cols:
            num_steps.append(("scaler", StandardScaler()))

        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        if num_cols:
            num_steps.insert(0, ("imputer", SimpleImputer(strategy="median")))
        cat_steps = []
        if cat_cols:
            cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]

        transformers = []
        if num_cols:
            from sklearn.pipeline import Pipeline as SKPipeline
            transformers.append(("num", SKPipeline(num_steps), num_cols))
        if cat_cols:
            from sklearn.pipeline import Pipeline as SKPipeline
            transformers.append(("cat", SKPipeline(cat_steps), cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

        for col in cat_cols:
            types = set(map(type, X[col].dropna().unique()))
            if len(types) > 1:
                st.warning(f"La colonne '{col}' mélange plusieurs types {types}. Conversion en string.")
                X[col] = X[col].astype(str)

        # choose model
        if model_choice == "auto":
            if task == "classification":
                model = RandomForestClassifier(random_state=int(random_state))
            else:
                model = RandomForestRegressor(random_state=int(random_state))
        elif model_choice == "random_forest":
            model = RandomForestClassifier(random_state=int(random_state)) if task=="classification" else RandomForestRegressor(random_state=int(random_state))
        else:
            model = LogisticRegression(max_iter=1000) if task=="classification" else LinearRegression()

        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # metrics
        if task == "classification":
            metrics = classification_metrics(y_test, preds)
        else:
            metrics = regression_metrics(y_test, preds)

        st.write("Metrics (test):", metrics)
        # save model
        model_path = f"outputs/models/model_{target}.pkl"
        helpers.ensure_dir("outputs/models")
        joblib.dump(pipe, model_path)
        st.success(f"Modèle entrainé et sauvegardé : {model_path}")

        return pipe, X_test, X_train, y_test, y_train

    st.stop()
