"""Train, compare, tune, evaluate and save the churn model.

Run from the project root:
    python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Workflow (no test-set peeking):
 1. split train/test (stratified)
 2. compare candidate models with 5-fold CV on the TRAINING set only
 3. tune the best candidate with RandomizedSearchCV (CV on training set)
 4. evaluate ONCE on the held-out test set and save everything
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV, StratifiedKFold, cross_validate,
)
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import (
    evaluate_pipeline, plot_confusion_matrix, plot_roc_curves, report,
    threshold_table,
)
from src.preprocessing import (
    RANDOM_STATE, build_pipeline, clean_data, load_data, make_split, split_xy,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # XGBoost is optional
    HAS_XGB = False

DECISION_THRESHOLD = 0.5  # default; see threshold_table for alternatives
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                 "f1": "F1", "roc_auc": "ROC-AUC"}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
           "f1": "f1", "roc_auc": "roc_auc"}


def get_candidates(y_train) -> dict:
    """Candidate models and their search spaces.

    class_weight / scale_pos_weight make the model pay more attention to the
    minority (churn) class. This usually raises recall at the cost of precision.
    """
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE),
            {"model__C": uniform(0.01, 10)},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"model__max_depth": randint(3, 10),
             "model__min_samples_leaf": randint(5, 50)},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                   n_jobs=-1),
            {"model__n_estimators": randint(100, 400),
             "model__max_depth": randint(4, 15),
             "model__min_samples_split": randint(2, 20),
             "model__min_samples_leaf": randint(1, 20),
             "model__max_features": ["sqrt", "log2"]},
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", scale_pos_weight=neg / pos,
                          random_state=RANDOM_STATE, n_jobs=-1),
            {"model__n_estimators": randint(100, 400),
             "model__max_depth": randint(2, 6),
             "model__learning_rate": uniform(0.01, 0.2),
             "model__subsample": uniform(0.7, 0.3)},
        )
    return candidates


def main(data_path: str, out_dir: str = ".", n_iter: int = 25) -> None:
    out = Path(out_dir)
    (out / "models").mkdir(exist_ok=True)
    (out / "reports" / "figures").mkdir(parents=True, exist_ok=True)

    # 1. Load, clean, split ---------------------------------------------------
    df = clean_data(load_data(data_path))
    X, y = split_xy(df)
    print(f"Rows: {len(df)} | churn rate: {y.mean():.3f}")
    X_train, X_test, y_train, y_test = make_split(X, y)

    # 2. Compare candidates with CV on the training set ---------------------
    candidates = get_candidates(y_train)
    cv_rows = []
    for name, (model, _) in candidates.items():
        scores = cross_validate(build_pipeline(model), X_train, y_train,
                                cv=CV, scoring=SCORING, n_jobs=1)
        cv_rows.append({"Model": name, **{
            label: scores[f"test_{key}"].mean() for key, label in METRIC_LABELS.items()}})
        print(f"CV done: {name}")
    cv_table = pd.DataFrame(cv_rows).set_index("Model").round(4)
    print("\nCross-validated comparison (training data only):")
    print(cv_table)

    # 3. Tune the best candidate by CV ROC-AUC -----------------------------
    best_name = cv_table["ROC-AUC"].idxmax()
    print(f"\nTuning: {best_name}")
    model, space = candidates[best_name]
    search = RandomizedSearchCV(
        build_pipeline(model), space, n_iter=n_iter, scoring="roc_auc",
        cv=CV, random_state=RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    final_pipeline = search.best_estimator_

    # 4. One-time evaluation on the test set --------------------------------
    test_rows, probs = [], {}
    for name, (m, _) in candidates.items():
        fitted = build_pipeline(m).fit(X_train, y_train)
        test_rows.append(evaluate_pipeline(name, fitted, X_test, y_test, DECISION_THRESHOLD))
        probs[name] = fitted.predict_proba(X_test)[:, 1]
    test_rows.append(evaluate_pipeline(f"{best_name} (tuned)", final_pipeline,
                                       X_test, y_test, DECISION_THRESHOLD))
    probs[f"{best_name} (tuned)"] = final_pipeline.predict_proba(X_test)[:, 1]
    test_table = pd.DataFrame(test_rows).set_index("Model").round(4)
    print("\nTest-set comparison:")
    print(test_table)

    y_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)
    print("\nClassification report (final model):")
    print(report(y_test, y_pred))
    thr = threshold_table(y_test, y_prob).round(3)
    print("Threshold trade-off:")
    print(thr.to_string(index=False))

    # 5. Save artefacts -------------------------------------------------------
    plot_confusion_matrix(y_test, y_pred, out / "reports/figures/confusion_matrix.png")
    plot_roc_curves(probs, y_test, out / "reports/figures/roc_curves.png")
    plt.close("all")

    joblib.dump(final_pipeline, out / "models/churn_model.pkl")

    # Small background sample (already transformed) used by SHAP in the app.
    sample = X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_STATE)
    feats = final_pipeline.named_steps["features"].transform(sample)
    bg = final_pipeline.named_steps["preprocessor"].transform(feats)
    bg = bg.toarray() if hasattr(bg, "toarray") else np.asarray(bg)
    joblib.dump(bg, out / "models/shap_background.pkl")
    # Raw sample for the global SHAP chart in the app / notebook.
    sample.to_csv(out / "models/sample_raw.csv", index=False)

    cv_table.to_csv(out / "reports/cv_comparison.csv")
    test_table.to_csv(out / "reports/test_comparison.csv")
    (out / "reports/test_comparison.md").write_text(test_table.to_markdown())
    (out / "reports/final_model.json").write_text(json.dumps({
        "selected_model": best_name,
        "best_params": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "cv_roc_auc": float(search.best_score_),
        "decision_threshold": DECISION_THRESHOLD,
        "test_metrics": test_table.loc[f"{best_name} (tuned)"].to_dict(),
    }, indent=2))
    print("\nSaved model to models/churn_model.pkl and reports to reports/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--out", default=".")
    parser.add_argument("--n-iter", type=int, default=25)
    args = parser.parse_args()
    main(args.data, args.out, args.n_iter)
