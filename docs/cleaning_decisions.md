# Data Cleaning Decisions (Milestone 2): CLOSED

Source file: `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`
Produced by: `python -m src.data_quality` (full output in `reports/data_quality_report.txt`)

## Confirmed findings (from running the code on the real file)

| Check | Result |
|---|---|
| Shape | 7,043 rows x 21 columns |
| Nulls reported by pandas | 0 |
| Hidden blank cells | 11 (all in `TotalCharges`) |
| `TotalCharges` after `pd.to_numeric(errors="coerce")` | 11 missing values |
| `tenure` of those 11 rows | all equal to 0 |
| Duplicate rows | 0 |
| Duplicate `customerID` values | 0 |
| Rows identical in every column except `customerID` | 22 (kept; see below) |
| Rows with `tenure == 0` | 11 |
| `tenure == 0` rows that have a non-blank `TotalCharges` | 0 |
| Blank `TotalCharges` AND `tenure == 0` | 11 (exact overlap) |
| Non-blank values that failed numeric conversion | 0 |
| Churn label of the 11 blank rows | all "No" |
| Negative tenure / MonthlyCharges <= 0 / TotalCharges < 0 | 0 / 0 / 0 |
| Category values | all valid (no misspellings or stray categories) |
| Phone/internet logic conflicts (e.g. "No phone service" with multiple lines) | 0 violations |
| After `clean_data()` | 7,043 x 20; `customerID` dropped; `TotalCharges` float64 with 11 NaN; `Churn` in {0, 1} |
| Churn = No | 5,174 (73.46%) |
| Churn = Yes | 1,869 (26.54%) |

Why pandas reported 0 nulls: the 11 cells contain a blank string (whitespace), not a real `NaN`. That string forces the whole `TotalCharges` column to be read as text (`object`).

Derived from the counts above (arithmetic only): the majority:minority ratio is 5,174 / 1,869 = about 2.8 : 1, and a model that always predicted "No" would be 73.46% accurate while catching 0 churners. This is why accuracy alone is not used to choose a model.

## Cleaning applied, and why

| # | Action | Where | Justification |
|---|---|---|---|
| 1 | Drop `customerID` | `clean_data()` | Unique identifier with no predictive meaning; keeping it would let models memorise rows. Verified unique (0 duplicates). |
| 2 | Convert `TotalCharges` to numeric with `errors="coerce"` | `clean_data()` | Only the 11 blank cells become `NaN`; no other text was silently destroyed (11 blanks = 11 NaNs). |
| 3 | Map `Churn` Yes/No to 1/0 | `clean_data()` | Models need a numeric target. 1 = churned. |
| 4 | Set `TotalCharges = 0` where it is missing AND `tenure == 0` | `FeatureEngineer.transform()` | All 11 blanks have `tenure == 0`: these customers had not been billed yet, so 0 is the logically correct value, not a guess. This is a fixed rule, not learned from data, so it cannot leak. |
| 5 | Median-impute any *other* missing numeric value; most-frequent for categoricals | `SimpleImputer` inside the pipeline | None exist in this file, but the saved model must survive missing values from app/user input. Fitted on training data only. |

## What was deliberately NOT done

- **The 11 rows were not dropped.** They are real customers, they are 0.16% of the data, and the Streamlit app lets users enter `tenure = 0`, so the model must handle that case.
- **No median/mean fill before the split.** Computing a median on the full dataset would let test-set information influence training (data leakage).
- **No de-duplication by profile.** 22 rows are identical in every column except `customerID`. With 19 mostly categorical features and rounded prices, different real customers can share a profile, and there are no fully duplicated rows (every `customerID` is unique). Dropping them would remove real customers. Because they are so few (22 of 7,043), keeping them has negligible effect either way.
- **No outlier removal.** No impossible values have been reported, and removing real customers without a reason would bias results.

## Note on the 11 zero-tenure customers
All 11 are labelled `Churn = No`. With only 11 rows this is **not** treated as an insight about churn: it is a property of how these brand-new, not-yet-billed customers appear in the data. The model should not be read as "tenure 0 means safe".

## Status
Milestone 2 is closed. Every check above was produced by `python -m src.data_quality` on the real file. Regression tests in `tests/test_pipeline.py` encode these facts.

## Leakage note
Everything before the split is either row-wise (drop ID, type conversion, label mapping) or a fixed domain rule (tenure 0 gives TotalCharges 0). Nothing learns a statistic from the data until the pipeline's `fit()` runs on training data only.
