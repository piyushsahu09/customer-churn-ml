"""Metrics and plots for classification models."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def evaluate_pipeline(name, pipeline, X_test, y_test, threshold=0.5) -> dict:
    """Evaluate on held-out data. The decision threshold is explicit."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return {"Model": name, **compute_metrics(y_test, y_pred, y_prob)}


def plot_confusion_matrix(y_true, y_pred, path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Stayed (0)", "Churned (1)"])
    ax.set_yticks([0, 1], ["Stayed (0)", "Churned (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_roc_curves(results: dict, y_test, path=None):
    """results: {model_name: predicted probabilities for class 1}"""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def threshold_table(y_true, y_prob, thresholds=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7)) -> pd.DataFrame:
    """Show the precision/recall trade-off at different decision thresholds."""
    rows = []
    for t in thresholds:
        pred = (np.asarray(y_prob) >= t).astype(int)
        rows.append({
            "Threshold": t,
            "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred),
            "F1": f1_score(y_true, pred),
            "Flagged customers": int(pred.sum()),
        })
    return pd.DataFrame(rows)


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=["Stayed", "Churned"])
