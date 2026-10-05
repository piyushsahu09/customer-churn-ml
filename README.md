# Customer Churn Prediction & Retention System

An end-to-end machine learning project that predicts which telecom customers are likely to churn, explains predictions with SHAP, assigns Low/Medium/High risk, and suggests rule-based retention actions through a Streamlit app.

## Business Problem

Customer churn directly affects recurring revenue. The goal is to identify customers who are more likely to leave so a retention team can prioritise outreach. The model produces a churn probability that can be ranked or thresholded according to the business cost of contacting customers.

## Dataset

IBM Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`), with 7,043 customers and 21 original columns.

| Group | Columns |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Account | tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Target | **Churn** (Yes = left, No = stayed) |

`customerID` is an identifier and is dropped. `TotalCharges` is converted from text to numeric. The 11 blank `TotalCharges` values in the dataset occur for customers with `tenure == 0`; the cleaning logic treats those as 0 because those customers have not yet been billed.

The observed churn rate is **26.54%** (1,869 churned and 5,174 stayed), so a majority-class baseline would achieve about **73.46% accuracy while identifying zero churners**. This is why recall, F1 and ROC-AUC are important alongside accuracy.

## Approach

```text
Data Validation -> Cleaning -> EDA -> Feature Engineering -> Stratified Split
-> Preprocessing Pipeline -> 5-fold CV Model Comparison -> Tuning
-> Held-out Test Evaluation -> SHAP -> Risk Segmentation
-> Rule-based Recommendations -> Streamlit App
```

Key design decisions:

- Split before fitting anything and keep imputation, scaling and encoding inside the scikit-learn pipeline to avoid data leakage.
- Feature engineering is part of the saved pipeline, so the Streamlit app accepts raw customer inputs using the same transformations used during training.
- Candidate models are compared with 5-fold cross-validation on the training set only; the held-out test set is evaluated once.
- Class imbalance is handled with `class_weight="balanced"` for Logistic Regression, Decision Tree and Random Forest, and `scale_pos_weight` for XGBoost.
- The default classification threshold is 0.5. A threshold table is also generated to show the precision/recall trade-off.
- Risk bands are project-defined: LOW < 0.30, MEDIUM 0.30-0.70, HIGH > 0.70.
- Retention recommendations are rule-based business heuristics and are kept separate from the ML model; they are not causal conclusions.

Engineered features include `tenure_group`, `avg_monthly_revenue`, `service_count`, and `has_protection_support`.

## Models

The project compares:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The model with the highest cross-validated ROC-AUC is tuned with `RandomizedSearchCV`. In the completed run, **Logistic Regression** was selected.

## Results

### Cross-validated comparison on training data

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7476 | 0.5157 | 0.7940 | 0.6251 | **0.8457** |
| Decision Tree | 0.7368 | 0.5038 | 0.5003 | 0.5019 | 0.6614 |
| Random Forest | 0.7733 | 0.5639 | 0.6441 | 0.6012 | 0.8261 |
| XGBoost | 0.7719 | 0.5597 | 0.6589 | 0.6050 | 0.8218 |

### Held-out test comparison

| Model | Accuracy | Precision | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.7374 | 0.5034 | 0.6138 | 0.8423 |
| Decision Tree | 0.7339 | 0.4987 | 0.5072 | 0.6646 |
| Random Forest | 0.7559 | 0.5350 | 0.5711 | 0.8162 |
| XGBoost | 0.7523 | 0.5278 | 0.5759 | 0.8165 |
| **Logistic Regression (tuned)** | **0.7410** | **0.5077** | **0.6178** | **0.8415** |

The final tuned model's verified recall is **0.7890**. The complete test metrics are also stored by the training script in `reports/test_comparison.csv` and `reports/final_model.json`.

### Final model

- Selected model: **Logistic Regression**
- Tuning method: `RandomizedSearchCV`, 5-fold CV, ROC-AUC scoring
- Best `model__C`: **6.128528947223795**
- Best CV ROC-AUC: **0.8458**
- Held-out test ROC-AUC: **0.8415**
- Held-out test recall: **0.7890**
- Held-out test F1: **0.6178**
- Decision threshold: **0.5**

The small CV-to-test ROC-AUC difference (0.8458 vs 0.8415) suggests the selected model generalised reasonably well to the held-out test set.

## EDA Findings

The EDA was run on the training split to avoid using the test set for exploratory decisions. Key observed associations were:

1. **Contract type:** month-to-month customers had a 42.75% churn rate, compared with 11.08% for one-year contracts and 2.87% for two-year contracts.
2. **Tenure:** churned customers had a median tenure of 10 months versus 38 months for customers who stayed.
3. **Monthly charges:** churned customers had a higher mean MonthlyCharges (74.86) than customers who stayed (61.34).
4. **Internet service:** Fiber optic customers had a 42.09% churn rate, compared with 18.69% for DSL and 7.25% for customers with no internet service.
5. **Payment method:** electronic-check customers had a 45.74% churn rate, the highest among the four payment methods.

These are **associations in the historical dataset, not causal effects**.

## Explainability

The Streamlit app provides:

- Local SHAP explanations for an individual customer.
- Global feature importance using mean absolute SHAP values over a training sample.
- A clear separation between model explanation and business-rule recommendations.

The final selected model is Logistic Regression, so the SHAP implementation uses `LinearExplainer` for that model. The code also supports `TreeExplainer` for tree-based models.

Do not interpret SHAP importance as proof that changing a feature will cause churn to increase or decrease.

## Streamlit App

The app accepts raw customer information, runs the saved pipeline, displays churn probability and risk level, shows SHAP contributions, and generates retention recommendations.

Local testing completed successfully. The app was **not yet deployed to Streamlit Community Cloud**, so the project does not claim a public cloud deployment.

Run locally:

```bash
python -m streamlit run app.py
```

## Installation

```bash
git clone <repository-url>
cd customer-churn-ml
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Train the model:

```bash
python -m src.train --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

This creates the saved pipeline and evaluation artefacts under `models/` and `reports/`.

Run the app:

```bash
python -m streamlit run app.py
```

Run tests:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The completed local test run passed **16/16 tests**.

## Evaluation Figures

Training/evaluation generates:

- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curves.png`
- `reports/test_comparison.md`
- `reports/final_model.json`

EDA generates eight figures under `reports/figures/eda/`.

## Data Leakage

Leakage occurs when information outside the training data influences model fitting. For example, fitting a scaler on the complete dataset before splitting would allow test-set information into training.

This project splits the data first and places preprocessing inside the pipeline, so preprocessing is fitted on the relevant training data during cross-validation and final training.

## Limitations

- The dataset is one historical snapshot from one telecom context; it does not provide a time series for robust drift analysis.
- SHAP explains model behaviour, not causality.
- Retention recommendations are heuristic rules and have not been validated with controlled experiments.
- The model's precision at the default threshold is about 0.51, so a production retention team would need to consider contact costs and choose a threshold accordingly.

## Future Improvements

- Probability calibration and cost-sensitive threshold selection.
- Model and data drift monitoring.
- Automated retraining and experiment tracking with MLflow.
- Additional behavioural features such as usage, support interactions and billing changes.
- API deployment with FastAPI.
- Uplift modelling to estimate which retention intervention is actually effective.

## Project Structure

```text
customer-churn-ml/
├── data/                  # dataset location
├── notebooks/             # analysis notebook
├── src/                   # preprocessing, training, evaluation, explainability, recommendations
├── models/                # saved pipeline and SHAP artefacts
├── reports/               # metrics tables and figures
├── tests/                 # automated tests
├── app.py                 # Streamlit application
├── requirements.txt
├── requirements-dev.txt
├── INTERVIEW_NOTES.md
└── README.md
```
