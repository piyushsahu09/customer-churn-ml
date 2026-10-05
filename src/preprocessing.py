"""Data loading, cleaning, feature engineering and the preprocessing pipeline.

Design choice: feature engineering lives INSIDE the sklearn pipeline
(``FeatureEngineer``). The saved model therefore accepts RAW customer rows
(exactly the columns of the original CSV) and applies every transformation
itself. This keeps the Streamlit app simple and consistent with training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Churn"
ID_COLUMN = "customerID"

# Raw input schema (what the Streamlit app must send).
RAW_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]

# Columns used to count subscribed services.
ADDON_SERVICES = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# Feature types AFTER feature engineering.
NUMERIC_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_revenue", "service_count", "has_protection_support",
]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group",
]


# ----------------------------------------------------------------- loading
def load_data(path: str) -> pd.DataFrame:
    """Read the raw CSV."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Light, row-wise cleaning that is safe to do before splitting.

    - drops the identifier column (no predictive meaning)
    - converts TotalCharges to numeric (blank strings become NaN)
    - maps the target to 0/1

    Nothing here learns from the data (no means/medians), so it cannot leak.
    Imputation happens later, inside the pipeline, after the split.
    """
    df = df.copy()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=ID_COLUMN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    return df


def split_xy(df: pd.DataFrame):
    """Separate features and target."""
    return df.drop(columns=TARGET), df[TARGET]


def make_split(X, y, test_size: float = 0.2):
    """THE train/test split used everywhere (training, EDA, notebook).

    stratify=y keeps the churn rate the same in both parts; a fixed
    random_state makes the split reproducible, so EDA and model training
    always see exactly the same training rows.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


# ------------------------------------------------------ feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Turns raw customer rows into model-ready columns (stateless).

    Engineered features:
    - tenure_group: 0-12, 13-24, 25-48, 49+ months
    - avg_monthly_revenue: TotalCharges / tenure (MonthlyCharges if tenure == 0)
    - service_count: number of subscribed services
    - has_protection_support: 1 if OnlineSecurity or TechSupport is "Yes"

    None of these use the target, so there is no target leakage.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if ID_COLUMN in df.columns:
            df = df.drop(columns=ID_COLUMN)

        for col in ("tenure", "MonthlyCharges", "TotalCharges"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # In the IBM data, TotalCharges is blank only when tenure == 0 (brand-new
        # customers not yet billed), so 0 is the logically correct value there.
        # Any other missing value is left for the pipeline's median imputer.
        df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

        # SeniorCitizen is a yes/no flag stored as 0/1: treat it as categorical.
        df["SeniorCitizen"] = (
            df["SeniorCitizen"].astype(str).replace({"0": "No", "1": "Yes"})
        )

        df["tenure_group"] = (
            pd.cut(df["tenure"], bins=[-1, 12, 24, 48, np.inf],
                   labels=["0-12", "13-24", "25-48", "49+"])
            .astype(object)
        )

        safe_tenure = df["tenure"].where(df["tenure"] > 0)
        df["avg_monthly_revenue"] = (df["TotalCharges"] / safe_tenure).fillna(
            df["MonthlyCharges"]
        )

        df["service_count"] = sum(
            (df[c] == "Yes").astype(int) for c in ADDON_SERVICES
        ) + (df["InternetService"] != "No").astype(int)

        df["has_protection_support"] = (
            (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
        ).astype(int)

        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# ------------------------------------------------------------- pipeline
def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns; impute + one-hot encode categoricals."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_pipeline(model) -> Pipeline:
    """Full pipeline: raw rows -> engineered features -> preprocessing -> model."""
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])
