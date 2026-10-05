"""Basic tests. Run from the project root:  pytest -q

Pipeline tests use synthetic data with the IBM schema, so they run even
without the real CSV. The data-loading test is skipped if the CSV is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from synthetic import make_synthetic_telco  # noqa: E402
from src.preprocessing import (  # noqa: E402
    RAW_FEATURES, build_pipeline, clean_data, load_data, split_xy,
)
from src.recommendations import recommend, risk_level  # noqa: E402

DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def fitted():
    df = clean_data(make_synthetic_telco(500))
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    return pipe, X_te


@pytest.mark.skipif(not DATA_PATH.exists(), reason="real dataset not present")
def test_data_loading():
    df = load_data(str(DATA_PATH))
    assert df.shape[0] > 1000
    assert {"Churn", "tenure", "TotalCharges", "Contract"} <= set(df.columns)


def test_clean_data_handles_blank_total_charges():
    df = clean_data(make_synthetic_telco(300))
    assert "customerID" not in df.columns
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().any()          # blanks became NaN
    assert set(df["Churn"].unique()) <= {0, 1}


def test_pipeline_handles_missing_values(fitted):
    pipe, X_te = fitted
    X = X_te.copy()
    X.iloc[0, X.columns.get_loc("MonthlyCharges")] = np.nan
    X.iloc[1, X.columns.get_loc("Contract")] = np.nan
    X.iloc[2, X.columns.get_loc("TotalCharges")] = np.nan
    prob = pipe.predict_proba(X)[:, 1]
    assert not np.isnan(prob).any()


def test_unseen_category_is_ignored(fitted):
    pipe, X_te = fitted
    X = X_te.head(3).copy()
    X["PaymentMethod"] = "Crypto"
    assert pipe.predict(X).shape == (3,)


def test_prediction_is_zero_or_one(fitted):
    pipe, X_te = fitted
    assert set(np.unique(pipe.predict(X_te))) <= {0, 1}


def test_probability_between_zero_and_one(fitted):
    pipe, X_te = fitted
    p = pipe.predict_proba(X_te)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_app_style_single_row_input(fitted):
    pipe, X_te = fitted
    row = X_te.iloc[[0]][RAW_FEATURES].copy()
    row["SeniorCitizen"] = 1            # app sends 0/1 ints
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1


def test_zero_tenure_does_not_break_features(fitted):
    pipe, X_te = fitted
    row = X_te.iloc[[0]].copy()
    row["tenure"] = 0
    row["TotalCharges"] = np.nan
    assert np.isfinite(pipe.predict_proba(row)[0, 1])


def test_risk_levels_and_configurable_thresholds():
    assert risk_level(0.1) == "LOW"
    assert risk_level(0.5) == "MEDIUM"
    assert risk_level(0.9) == "HIGH"
    assert risk_level(0.5, low=0.2, high=0.4) == "HIGH"


def test_recommendations_depend_on_risk():
    cust = {"Contract": "Month-to-month", "MonthlyCharges": 95, "InternetService": "Fiber optic",
            "TechSupport": "No", "OnlineSecurity": "No", "PaymentMethod": "Electronic check",
            "tenure": 3}
    assert "longer-term contract" in " ".join(recommend(cust, 0.85))
    assert "No retention action" in recommend(cust, 0.05)[0]


@pytest.mark.skipif(not (ROOT / "models" / "churn_model.pkl").exists(),
                    reason="trained model not present")
def test_streamlit_app_launches():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception


# ---------------------------------------------------------------------------
# Milestone 2 regression tests
# ---------------------------------------------------------------------------
def test_only_zero_tenure_blanks_are_filled_with_zero():
    """Blank TotalCharges (tenure==0) becomes 0; nothing else is invented."""
    from src.preprocessing import FeatureEngineer
    raw = make_synthetic_telco(400)
    blank = raw["TotalCharges"].str.strip() == ""
    assert (raw.loc[blank, "tenure"] == 0).all()          # the dataset quirk
    out = FeatureEngineer().transform(raw.drop(columns="Churn"))
    assert out["TotalCharges"].isna().sum() == 0
    assert (out.loc[blank.values, "TotalCharges"] == 0).all()


def test_nonzero_tenure_missing_total_is_left_for_imputer():
    from src.preprocessing import FeatureEngineer
    raw = make_synthetic_telco(50).drop(columns="Churn")
    raw.loc[raw["tenure"] > 0, "TotalCharges"] = raw.loc[raw["tenure"] > 0, "TotalCharges"].astype(object)
    idx = raw.index[raw["tenure"] > 0][0]
    raw.loc[idx, "TotalCharges"] = " "
    out = FeatureEngineer().transform(raw)
    assert np.isnan(out.loc[idx, "TotalCharges"])         # not silently zeroed


@pytest.mark.skipif(not DATA_PATH.exists(), reason="real dataset not present")
def test_real_dataset_matches_verified_milestone2_findings():
    """Facts verified by running src.data_quality on the real file."""
    raw = load_data(str(DATA_PATH))
    assert raw.shape == (7043, 21)
    assert raw.duplicated().sum() == 0
    assert raw["customerID"].duplicated().sum() == 0
    blank = raw["TotalCharges"].astype(str).str.strip() == ""
    assert blank.sum() == 11
    assert (raw.loc[blank, "tenure"] == 0).all()
    counts = raw["Churn"].value_counts()
    assert counts["No"] == 5174 and counts["Yes"] == 1869
    cleaned = clean_data(raw)
    assert cleaned["TotalCharges"].isna().sum() == 11
    assert "customerID" not in cleaned.columns
    assert cleaned.shape == (7043, 20)
    assert str(cleaned["TotalCharges"].dtype) == "float64"
    assert set(cleaned["Churn"].unique()) == {0, 1}
    # extra facts verified in the final Milestone 2 report
    assert (raw["tenure"] == 0).sum() == 11
    assert ((raw["tenure"] == 0) & ~blank).sum() == 0
    assert (raw.loc[blank, "Churn"] == "No").all()
    assert raw.drop(columns="customerID").duplicated().sum() == 22
    assert (raw["tenure"] < 0).sum() == 0
    assert (raw["MonthlyCharges"] <= 0).sum() == 0


# ---------------------------------------------------------------------------
# Milestone 3 tests
# ---------------------------------------------------------------------------
def test_make_split_is_reproducible_and_stratified():
    from src.preprocessing import make_split
    X, y = split_xy(clean_data(make_synthetic_telco(500)))
    a = make_split(X, y)
    b = make_split(X, y)
    assert a[0].index.equals(b[0].index)                  # same rows every time
    assert abs(a[2].mean() - a[3].mean()) < 0.01          # churn rate preserved
    assert len(a[1]) == round(0.2 * len(X))


def test_eda_steps_run_and_use_training_split_only(tmp_path):
    from src import eda
    csv = tmp_path / "telco.csv"
    make_synthetic_telco(400).to_csv(csv, index=False)
    frame = eda.load_eda_frame(csv)
    assert len(frame) == 320                              # 80% of 400, not all rows
    full = eda.load_eda_frame(csv, use_full=True)
    assert len(full) == 400
    for step in sorted(eda.STEPS):
        eda.run_step(step, csv, tmp_path / "figs")
    assert len(list((tmp_path / "figs").glob("*.png"))) == 8
