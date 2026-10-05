"""SHAP explanations for the final pipeline.

SHAP assigns each feature a contribution to one prediction (relative to a
baseline). It describes MODEL behaviour, not real-world causation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

_CATEGORICAL_PREFIXES = [
    "tenure_group", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "PaperlessBilling", "PaymentMethod", "PhoneService",
    "SeniorCitizen", "Dependents", "Partner", "Contract", "gender",
]


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def prettify(name: str) -> str:
    """'cat__Contract_Month-to-month' -> 'Contract = Month-to-month'."""
    name = name.split("__", 1)[-1]
    for col in _CATEGORICAL_PREFIXES:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def transform_features(pipeline, X_raw: pd.DataFrame):
    """Raw rows -> dense matrix the model actually sees + readable names."""
    feats = pipeline.named_steps["features"].transform(X_raw)
    prep = pipeline.named_steps["preprocessor"]
    matrix = _dense(prep.transform(feats))
    names = [prettify(n) for n in prep.get_feature_names_out()]
    return matrix, names


def build_explainer(pipeline, background: np.ndarray):
    """Pick the right explainer for the model type."""
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):  # linear models (Logistic Regression)
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)  # RandomForest / XGBoost / trees


def _positive_class(values):
    """Different SHAP/sklearn versions return different shapes; keep class 1."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def shap_values(explainer, matrix: np.ndarray) -> np.ndarray:
    return _positive_class(explainer.shap_values(matrix))


def explain_customer(pipeline, explainer, X_raw: pd.DataFrame, top_n: int = 5):
    """Return (increasing_risk, decreasing_risk) as lists of (feature, value)."""
    matrix, names = transform_features(pipeline, X_raw)
    values = shap_values(explainer, matrix)[0]
    contrib = pd.Series(values, index=names).sort_values()
    decreasing = [(k, v) for k, v in contrib.head(top_n).items() if v < 0]
    increasing = [(k, v) for k, v in contrib.tail(top_n)[::-1].items() if v > 0]
    return increasing, decreasing


def global_importance(pipeline, explainer, X_raw: pd.DataFrame) -> pd.Series:
    """Mean |SHAP| per feature over a sample of raw rows."""
    matrix, names = transform_features(pipeline, X_raw)
    values = shap_values(explainer, matrix)
    return pd.Series(np.abs(values).mean(axis=0), index=names).sort_values(ascending=False)
