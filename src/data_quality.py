"""Milestone 2: data-quality checks on the raw Telco CSV.

Run from the project root:
    python -m src.data_quality
    python -m src.data_quality --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv

Nothing here changes your data on disk. Everything is printed AND saved to
reports/data_quality_report.txt so you can paste it back or cite it later.
All numbers come from the file you give it; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import ID_COLUMN, TARGET, clean_data, load_data

PLACEHOLDERS = {"", "na", "n/a", "nan", "null", "none", "?", "-", "--", "unknown"}
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies"]

ROOT = Path(__file__).resolve().parents[1]      # the customer-churn-ml folder
DEFAULT_DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_REPORT = ROOT / "reports" / "data_quality_report.txt"
_report_file: Path | None = None


def out(text: str = "") -> None:
    """Print AND append to the report file immediately, so a crash later in
    the script still leaves everything printed so far on disk."""
    print(text)
    if _report_file is not None:
        with open(_report_file, "a", encoding="utf-8") as f:
            f.write(str(text) + "\n")


def section(title: str) -> None:
    out("\n" + "=" * 70)
    out(title)
    out("=" * 70)


def main(data_path, report_path) -> None:
    global _report_file
    _report_file = Path(report_path)
    _report_file.parent.mkdir(parents=True, exist_ok=True)
    _report_file.write_text("", encoding="utf-8")        # start a fresh report
    print(f"Reading data from: {Path(data_path).resolve()}")
    print(f"Saving report to:  {_report_file.resolve()}\n")

    df = load_data(str(data_path))
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    # ------------------------------------------------------------------ 1
    section("1. BASIC STRUCTURE")
    out(f"Shape (rows, columns): {df.shape}")
    out("\nData types:")
    out(df.dtypes.to_string())

    # ------------------------------------------------------------------ 2
    section("2. MISSING VALUES: what pandas reports")
    nulls = df.isnull().sum()
    out(f"Total null cells reported by pandas: {int(nulls.sum())}")
    out(nulls[nulls > 0].to_string() if nulls.sum() else "(no columns with nulls)")

    # ------------------------------------------------------------------ 3
    section("3. HIDDEN MISSING VALUES: blank / whitespace / placeholder text")
    out("Pandas only counts real NaN as null. A cell containing ' ' is NOT null.")
    found_any = False
    for col in text_cols:
        s = df[col].astype(str)
        blank = s.str.strip() == ""
        placeholder = s.str.strip().str.lower().isin(PLACEHOLDERS - {""}) 
        padded = (s != s.str.strip()) & ~blank
        if blank.sum() or placeholder.sum() or padded.sum():
            found_any = True
            out(f"  {col}: blank/whitespace-only={int(blank.sum())}, "
                f"placeholder-like={int(placeholder.sum())}, "
                f"leading/trailing spaces (non-blank)={int(padded.sum())}")
    if not found_any:
        out("  None found in any text column.")

    # ------------------------------------------------------------------ 4
    section("4. WHY IS TotalCharges NOT NUMERIC?")
    tc_raw = df["TotalCharges"]
    out(f"TotalCharges dtype: {tc_raw.dtype}")
    tc_blank = tc_raw.astype(str).str.strip() == ""
    tc_num = pd.to_numeric(tc_raw, errors="coerce")
    coerced_nan = tc_num.isna()
    out(f"Blank/whitespace TotalCharges cells: {int(tc_blank.sum())}")
    out(f"NaN after pd.to_numeric(errors='coerce'): {int(coerced_nan.sum())}")

    unexplained = coerced_nan & ~tc_blank
    out(f"Cells that FAILED to convert but are NOT blank: {int(unexplained.sum())}")
    if unexplained.sum():
        out("  -> These need manual inspection (unexpected text):")
        out(tc_raw[unexplained].value_counts().head(20).to_string())
    else:
        out("  -> Good: every conversion failure is a blank cell, nothing else.")

    if tc_blank.sum():
        cols = [c for c in [ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges", TARGET]
                if c in df.columns]
        out("\nRows with blank TotalCharges:")
        out(df.loc[tc_blank, cols].to_string())
        out(f"\nUnique tenure values among these rows: "
            f"{sorted(df.loc[tc_blank, 'tenure'].unique().tolist())}")
        out(f"Churn values among these rows: "
            f"{df.loc[tc_blank, TARGET].value_counts().to_dict()}")
    zero_tenure = df["tenure"] == 0
    out(f"\nRows with tenure == 0: {int(zero_tenure.sum())}")
    out(f"Of those, rows with a NON-blank TotalCharges: "
        f"{int((zero_tenure & ~tc_blank).sum())}")
    both = (tc_blank & zero_tenure).sum()
    out(f"Blank TotalCharges AND tenure == 0: {int(both)}")
    if tc_blank.sum() and both == tc_blank.sum() == zero_tenure.sum():
        out("  -> Blank TotalCharges <=> tenure == 0 exactly. Missingness has a clear "
            "cause (customer not billed yet), so 0 is a justified value.")
    elif tc_blank.sum():
        out("  -> The two do NOT line up exactly. Do not assume 0; tell me this output.")

    # ------------------------------------------------------------------ 5
    section("5. DUPLICATES")
    out(f"Fully duplicated rows: {int(df.duplicated().sum())}")
    if ID_COLUMN in df.columns:
        out(f"Duplicated customerID values: {int(df[ID_COLUMN].duplicated().sum())}")
        feats = df.drop(columns=ID_COLUMN)
        out(f"Rows identical in all columns EXCEPT customerID: "
            f"{int(feats.duplicated().sum())}  (informational only: different customers "
            f"can legitimately share the same profile, so we do NOT drop these)")

    # ------------------------------------------------------------------ 6
    section("6. INVALID / UNEXPECTED VALUES")
    out("Distinct values of each text column (small ones listed in full):")
    for col in text_cols:
        if col == ID_COLUMN:
            continue
        n = df[col].nunique()
        if col == "TotalCharges":
            continue
        if n <= 10:
            out(f"  {col} ({n}): {sorted(df[col].astype(str).unique().tolist())}")
        else:
            out(f"  {col}: {n} distinct values (unexpected for a categorical column)")

    out("\nNumeric ranges:")
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
        out(f"  {col}: min={df[col].min()}, max={df[col].max()}")
    out(f"  TotalCharges (valid numbers only): min={tc_num.min()}, max={tc_num.max()}")
    out(f"  SeniorCitizen values: {sorted(df['SeniorCitizen'].unique().tolist())}")
    out(f"  Negative tenure: {int((df['tenure'] < 0).sum())} | "
        f"MonthlyCharges <= 0: {int((df['MonthlyCharges'] <= 0).sum())} | "
        f"TotalCharges < 0: {int((tc_num < 0).sum())}")

    out("\nLogical consistency checks (0 violations expected):")
    v1 = (df["PhoneService"] == "No") & (df["MultipleLines"] != "No phone service")
    v2 = (df["PhoneService"] == "Yes") & (df["MultipleLines"] == "No phone service")
    out(f"  PhoneService=No but MultipleLines not 'No phone service': {int(v1.sum())}")
    out(f"  PhoneService=Yes but MultipleLines='No phone service':    {int(v2.sum())}")
    no_net = df["InternetService"] == "No"
    for col in ADDON_COLS:
        a = (no_net & (df[col] != "No internet service")).sum()
        b = (~no_net & (df[col] == "No internet service")).sum()
        out(f"  {col}: InternetService=No mismatches={int(a)}, "
            f"has-internet-but-'No internet service'={int(b)}")

    ratio = (tc_num / (df["tenure"].where(df["tenure"] > 0) * df["MonthlyCharges"]))
    out("\nTotalCharges / (tenure x MonthlyCharges), informational (not a rule: "
        "prices change over time):")
    out(ratio.describe().round(3).to_string())

    # ------------------------------------------------------------------ 7
    section("7. TARGET DISTRIBUTION (raw)")
    out(df[TARGET].value_counts().to_string())
    out(df[TARGET].value_counts(normalize=True).round(4).to_string())

    # ------------------------------------------------------------------ 8
    section("8. RESULT OF clean_data() (the function the pipeline uses)")
    cleaned = clean_data(df)
    out(f"Shape after cleaning: {cleaned.shape}")
    out(f"customerID dropped: {ID_COLUMN not in cleaned.columns}")
    out(f"TotalCharges dtype: {cleaned['TotalCharges'].dtype}")
    out(f"TotalCharges NaN remaining: {int(cleaned['TotalCharges'].isna().sum())}")
    out(f"Churn values after mapping: {sorted(cleaned[TARGET].unique().tolist())}")
    out(f"Churn rate (share of 1s): {cleaned[TARGET].mean():.4f}")
    out("\nNOTE: the remaining NaNs are deliberately NOT filled here. The model "
        "pipeline sets TotalCharges=0 only where tenure==0 (justified above) and "
        "uses a median imputer, fitted on TRAINING data only, for anything else. "
        "Filling with a median now, before the split, would leak information.")

    print(f"\nDONE. Report saved to: {_report_file.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    args = ap.parse_args()
    main(args.data, args.report)
