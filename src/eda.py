"""Milestone 3: exploratory data analysis, ONE chart per step.

Run from anywhere:
    python -m src.eda --step 1          # churn distribution
    python -m src.eda --step 2          # churn by contract
    ... up to --step 8

Each step prints a numeric table (the evidence) and saves a PNG to
reports/figures/eda/. The script never writes interpretations: those are
yours to write in reports/eda_notes.md after looking at the numbers/chart.

By default EDA uses the TRAINING split only (the same rows the model trains
on). This stops patterns in the test rows from influencing which features we
build. Use --full to explore all 7,043 rows (for curiosity, not for decisions).

Charts show ASSOCIATIONS in this dataset, not causes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save files; don't open windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.preprocessing import TARGET, clean_data, load_data, make_split, split_xy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG_DIR = ROOT / "reports" / "figures" / "eda"

LABELS = {0: "Stayed", 1: "Churned"}
ORDER = ["Stayed", "Churned"]
PALETTE = {"Stayed": "#4C72B0", "Churned": "#C44E52"}
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200)        # never truncate tables with "..."
pd.set_option("display.max_columns", 50)


# ----------------------------------------------------------------- helpers
def load_eda_frame(data_path, use_full: bool = False) -> pd.DataFrame:
    """Cleaned data (TotalCharges numeric, Churn 0/1) + a readable label column."""
    df = clean_data(load_data(str(data_path)))
    if use_full:
        frame = df
    else:
        X, y = split_xy(df)
        X_train, _, y_train, _ = make_split(X, y)
        frame = X_train.assign(**{TARGET: y_train})
    frame = frame.copy()
    frame["Churn_label"] = frame[TARGET].map(LABELS)
    return frame


def rate_table(frame: pd.DataFrame, col: str, order=None) -> pd.DataFrame:
    """Customers, churners and churn rate for each category of `col`."""
    t = frame.groupby(col)[TARGET].agg(customers="count", churned="sum")
    t["churn_rate_%"] = (100 * t["churned"] / t["customers"]).round(2)
    t["share_of_customers_%"] = (100 * t["customers"] / len(frame)).round(2)
    return t.loc[order] if order is not None else t.sort_index()


def describe_by_churn(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    return frame.groupby("Churn_label")[col].describe().loc[ORDER].round(2)


def save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def header(step: int, title: str, frame: pd.DataFrame, use_full: bool) -> None:
    scope = "ALL rows" if use_full else "TRAINING split only"
    print("=" * 70)
    print(f"EDA step {step}: {title}")
    print(f"Data used: {scope} ({len(frame)} customers, "
          f"overall churn rate {100 * frame[TARGET].mean():.2f}%)")
    print("=" * 70)


def footer(step: int, path: Path) -> None:
    print(f"\nChart saved to: {path}")
    print(f"NEXT: open the chart, then write YOUR interpretation under "
          f"'Step {step}' in reports/eda_notes.md")


# ------------------------------------------------------------------- steps
def step1_churn_distribution(frame, out_dir, use_full=False):
    header(1, "Churn distribution", frame, use_full)
    counts = frame["Churn_label"].value_counts().reindex(ORDER)
    pct = (100 * counts / counts.sum()).round(2)
    print(pd.DataFrame({"customers": counts, "percent": pct}))
    print(f"\nMajority : minority ratio = {counts['Stayed'] / counts['Churned']:.2f} : 1")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,
                palette=PALETTE, legend=False, ax=ax)
    for i, (c, p) in enumerate(zip(counts.values, pct.values)):
        ax.text(i, c, f"{c}\n({p}%)", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.set_title("Churn distribution")
    ax.set_ylabel("Customers")
    ax.set_xlabel("")
    footer(1, save(fig, "01_churn_distribution.png", out_dir))


def _categorical_step(step, title, col, filename, frame, out_dir, use_full, rotate=0):
    header(step, title, frame, use_full)
    table = rate_table(frame, col)
    print(table)
    overall = 100 * frame[TARGET].mean()
    print(f"\nOverall churn rate for comparison: {overall:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    order = table.index.tolist()
    sns.countplot(data=frame, x=col, hue="Churn_label", order=order,
                  hue_order=ORDER, palette=PALETTE, ax=axes[0])
    axes[0].set_title(f"Customers by {col} (counts)")
    axes[0].set_xlabel("")
    sns.barplot(x=table.index, y=table["churn_rate_%"], order=order,
                color="#8172B2", ax=axes[1])
    axes[1].axhline(overall, color="black", ls="--", lw=1,
                    label=f"overall {overall:.1f}%")
    for i, v in enumerate(table["churn_rate_%"]):
        axes[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    axes[1].set_title(f"Churn rate within each {col} group")
    axes[1].set_ylabel("Churn rate (%)")
    axes[1].set_xlabel("")
    axes[1].legend()
    for ax in axes:
        ax.tick_params(axis="x", rotation=rotate)
    footer(step, save(fig, filename, out_dir))


def step2_contract(frame, out_dir, use_full=False):
    _categorical_step(2, "Churn by Contract", "Contract", "02_churn_by_contract.png",
                      frame, out_dir, use_full)


def step3_tenure(frame, out_dir, use_full=False):
    header(3, "Churn by tenure (months with the company)", frame, use_full)
    print(describe_by_churn(frame, "tenure"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(data=frame, x="tenure", hue="Churn_label", hue_order=ORDER,
                 palette=PALETTE, bins=36, element="step", ax=axes[0])
    axes[0].set_title("Tenure distribution (counts)")
    sns.boxplot(data=frame, x="Churn_label", y="tenure", order=ORDER,
                hue="Churn_label", palette=PALETTE, legend=False, ax=axes[1])
    axes[1].set_title("Tenure by churn")
    axes[1].set_xlabel("")
    footer(3, save(fig, "03_tenure_vs_churn.png", out_dir))


def _boxplot_step(step, title, col, filename, frame, out_dir, use_full):
    header(step, title, frame, use_full)
    missing = int(frame[col].isna().sum())
    if missing:
        print(f"Note: {missing} row(s) have missing {col} and are left out of "
              f"this chart/table.\n")
    print(describe_by_churn(frame, col))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.boxplot(data=frame, x="Churn_label", y=col, order=ORDER,
                hue="Churn_label", palette=PALETTE, legend=False, ax=ax)
    ax.set_title(f"{col} by churn")
    ax.set_xlabel("")
    footer(step, save(fig, filename, out_dir))


def step4_monthly_charges(frame, out_dir, use_full=False):
    _boxplot_step(4, "Monthly charges vs churn", "MonthlyCharges",
                  "04_monthly_charges_vs_churn.png", frame, out_dir, use_full)


def step5_total_charges(frame, out_dir, use_full=False):
    _boxplot_step(5, "Total charges vs churn", "TotalCharges",
                  "05_total_charges_vs_churn.png", frame, out_dir, use_full)
    print("Reminder: TotalCharges grows with tenure, so it mixes 'how long' "
          "with 'how much per month'.")


def step6_internet_service(frame, out_dir, use_full=False):
    _categorical_step(6, "Churn by InternetService", "InternetService",
                      "06_churn_by_internet_service.png", frame, out_dir, use_full)


def step7_payment_method(frame, out_dir, use_full=False):
    _categorical_step(7, "Churn by PaymentMethod", "PaymentMethod",
                      "07_churn_by_payment_method.png", frame, out_dir, use_full,
                      rotate=20)


def step8_correlation(frame, out_dir, use_full=False):
    header(8, "Correlation matrix (numeric columns only)", frame, use_full)
    cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", TARGET]
    corr = frame[cols].corr().round(3)  # pandas skips missing values pairwise
    print(corr)
    print("\nNote: SeniorCitizen and Churn are 0/1 flags, so their correlations "
          "are rough summaries. Correlation measures LINEAR association only.")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation (numeric features)")
    footer(8, save(fig, "08_correlation_matrix.png", out_dir))


STEPS = {
    1: step1_churn_distribution, 2: step2_contract, 3: step3_tenure,
    4: step4_monthly_charges, 5: step5_total_charges,
    6: step6_internet_service, 7: step7_payment_method, 8: step8_correlation,
}


def run_step(step: int, data_path=DEFAULT_DATA, out_dir=FIG_DIR, use_full=False):
    frame = load_eda_frame(data_path, use_full)
    STEPS[step](frame, Path(out_dir), use_full)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, required=True, choices=sorted(STEPS))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--full", action="store_true",
                    help="use all rows instead of the training split")
    args = ap.parse_args()
    run_step(args.step, args.data, FIG_DIR, args.full)
