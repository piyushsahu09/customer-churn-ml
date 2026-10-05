# Interview Notes

These answers are based on the completed local run of the Customer Churn Prediction & Retention System. Use the verified numbers below rather than inventing or estimating metrics.

## The 30 questions

**1. Why churn prediction?**
It's a common, high-value business problem with a clear target, tabular data and an actionable output: a ranked list of at-risk customers for the retention team.

**2. Target variable?**
`Churn` (Yes = 1, No = 0). It is a binary classification problem. The dataset contains **1,869 churners out of 7,043 customers, a 26.54% churn rate**.

**3. What preprocessing did you do?**
I dropped `customerID`, converted `TotalCharges` to numeric, engineered features, then used a scikit-learn pipeline for numeric imputation/scaling and categorical imputation/one-hot encoding. The saved pipeline contains the feature engineering and preprocessing used by the app.

**4. How did you handle missing values?**
There were **11 blank `TotalCharges` values**. All 11 belonged to customers with `tenure == 0`, so the cleaning step sets those specific values to 0 because those customers had not yet been billed. Other missing values are handled by `SimpleImputer` inside the training pipeline.

**5. Why median imputation?**
The median is robust to outliers and skewed distributions, unlike the mean. It is also fitted within the pipeline so the imputation statistic comes from training data only.

**6. Why one-hot encoding?**
Most categorical variables, such as contract and payment method, have no natural numeric order. One-hot encoding avoids introducing a false ordering. `handle_unknown="ignore"` also helps the app handle an unseen category safely.

**7. Why scale numeric features?**
Logistic Regression is sensitive to feature scale, especially because of regularisation. Tree models do not require scaling, but using the same pipeline structure keeps preprocessing consistent across candidates.

**8. Why train/test split?**
To estimate performance on unseen data. Evaluating on the training data would reward memorisation rather than generalisation.

**9. What is data leakage?**
Data leakage is when information from outside the training data affects model fitting. For example, fitting a scaler on the entire dataset before splitting would leak test-set information. I split first and used a pipeline so preprocessing is fitted only on the appropriate training data.

**10. Why stratified splitting?**
It preserves the churn proportion in both train and test sets, which is useful because churn is an imbalanced target.

**11. Why Logistic Regression as the baseline?**
It is simple, fast and interpretable. A more complex model should have to demonstrate a meaningful improvement before it is preferred.

**12. Why compare multiple models?**
No single algorithm is best for every dataset. I compared Logistic Regression, Decision Tree, Random Forest and XGBoost using the same 5-fold CV setup on the training data.

**13. Why isn't accuracy enough?**
Only **26.54%** of customers churned. A model that predicts everyone as “stayed” would get **73.46% accuracy while catching zero churners**. That is why I also evaluated precision, recall, F1 and ROC-AUC.

**14. Precision vs recall?**
Precision is the proportion of flagged customers who actually churn. Recall is the proportion of actual churners that the model successfully flags.

**15. F1-score?**
F1 is the harmonic mean of precision and recall. It becomes low when either precision or recall is low.

**16. ROC-AUC?**
ROC-AUC measures ranking quality across classification thresholds. A score of 0.5 is roughly random ranking, while 1.0 is perfect separation.

**17. Cross-validation?**
I used 5-fold stratified cross-validation on the training set. The training data is divided into five folds; each fold is used once for validation while the other four are used for training, then the scores are averaged.

**18. Hyperparameter tuning?**
I used `RandomizedSearchCV` with 5-fold CV and ROC-AUC scoring. Logistic Regression was the best candidate by CV ROC-AUC, and its `C` parameter was tuned.

**19. How did you prevent overfitting?**
I used cross-validation for model selection, regularisation/model constraints, a held-out test set evaluated only after model selection, and compared CV ROC-AUC with test ROC-AUC. For the final tuned Logistic Regression, CV ROC-AUC was **0.8458** and test ROC-AUC was **0.8415**.

**20. Most important features?**
The app calculates global feature importance as **mean absolute SHAP value over a training sample**. The exact ranking should be read from the app's “Global feature importance” chart for the current saved model rather than memorised from an unverified list. The local example shown during testing is not a valid substitute for the global ranking.

**21. Does feature importance prove causation?**
No. It shows which features the model relies on for its predictions. For example, month-to-month customers have a higher observed churn rate, but that does not prove that changing someone's contract would cause them to stay.

**22. How does SHAP work?**
SHAP is based on Shapley values from game theory. It assigns feature contributions to a prediction relative to a baseline, and the contributions add up to the model output difference. For the final Logistic Regression model, the code uses `LinearExplainer`; the implementation also supports `TreeExplainer` for tree models.

**23. End-to-end flow?**
User enters raw customer fields in Streamlit -> the saved pipeline performs feature engineering and preprocessing -> the model produces churn probability -> the app assigns a risk band -> SHAP explains the prediction -> separate business rules generate retention recommendations.

**24. How did you deploy?**
I **tested the application locally with Streamlit** using the saved model pipeline. I have not yet deployed it to Streamlit Community Cloud, so I would not claim a public cloud deployment in an interview.

**25. How would you monitor in production?**
I would track input feature distributions, prediction distributions, model latency/errors and, once outcomes arrive, recall, precision and ROC-AUC on recent cohorts. I would also define alerts for meaningful drift or performance degradation.

**26. Customer behaviour changes?**
That can create concept drift or broader data drift. I would compare recent cohorts with the baseline, investigate which features or relationships changed, and retrain if performance deteriorates.

**27. Model drift handling?**
Use input drift tests such as PSI or KS, monitor performance when labels become available, and define a retraining/promotion trigger based on business and model metrics.

**28. Retraining?**
I would rerun the same training pipeline on fresh data, evaluate the candidate on a recent holdout, compare it with the current production model, and promote it only if it meets the predefined acceptance criteria.

**29. How would you improve it?**
I would add probability calibration, cost-sensitive threshold selection, more behavioural features such as usage/support history, experiment tracking with MLflow, an API layer, and uplift modelling to estimate which retention intervention actually works.

**30. If the business wanted higher recall?**
Lower the decision threshold so more potential churners are flagged, while accepting more false positives. In the completed run, the threshold table shows the trade-off: at threshold **0.2**, recall is **0.963** with **917** customers flagged; at **0.5**, recall is **0.789** with **581** customers flagged.

## Verified model results

Final selected model: **tuned Logistic Regression**

- Best tuned `C`: **6.128528947223795**
- Best CV ROC-AUC: **0.8458**
- Test accuracy: **0.7410**
- Test precision: **0.5077**
- Test recall: **0.7890**
- Test F1: **0.6178**
- Test ROC-AUC: **0.8415**
- Default decision threshold: **0.5**

### Cross-validated model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7476 | 0.5157 | 0.7940 | 0.6251 | **0.8457** |
| Decision Tree | 0.7368 | 0.5038 | 0.5003 | 0.5019 | 0.6614 |
| Random Forest | 0.7733 | 0.5639 | 0.6441 | 0.6012 | 0.8261 |
| XGBoost | 0.7719 | 0.5597 | 0.6589 | 0.6050 | 0.8218 |

## EDA facts worth remembering

These are associations from the training-split EDA, not causal conclusions:

- Overall churn: **26.54%**.
- Month-to-month contract churn: **42.75%**.
- One-year contract churn: **11.08%**.
- Two-year contract churn: **2.87%**.
- Median tenure: **10 months for churned vs 38 months for stayed**.
- Mean MonthlyCharges: **74.86 for churned vs 61.34 for stayed**.
- Fiber optic churn: **42.09%**; DSL: **18.69%**; no internet: **7.25%**.
- Electronic-check churn: **45.74%**.

## Resume bullets

- Built an end-to-end customer churn prediction system on the IBM Telco dataset covering **7,043 customers**, using scikit-learn pipelines with leakage-safe preprocessing, stratified splitting and 5-fold cross-validation.
- Compared Logistic Regression, Decision Tree, Random Forest and XGBoost; tuned Logistic Regression with `RandomizedSearchCV`, achieving **0.8415 ROC-AUC and 0.789 recall** on a held-out test set.
- Added SHAP-based global and per-customer explanations, configurable Low/Medium/High risk bands, and a separate rule-based retention recommendation layer.
- Built and locally tested an interactive Streamlit app that serves predictions from a single saved pipeline; **16/16 automated tests passed**.
