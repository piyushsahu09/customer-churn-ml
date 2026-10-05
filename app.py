"""Streamlit app: Customer Churn Prediction & Retention System.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.preprocessing import RAW_FEATURES
from src.recommendations import (
    HIGH_THRESHOLD, LOW_THRESHOLD, recommend, risk_level,
)

try:
    from src import explain
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

MODEL_PATH = Path("models/churn_model.pkl")
BACKGROUND_PATH = Path("models/shap_background.pkl")
SAMPLE_PATH = Path("models/sample_raw.csv")

st.set_page_config(page_title="Churn Prediction", page_icon="📉", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_explainer(_pipeline):
    if not HAS_SHAP or not BACKGROUND_PATH.exists():
        return None
    return explain.build_explainer(_pipeline, joblib.load(BACKGROUND_PATH))


st.title("Customer Churn Prediction & Retention System")
st.caption(
    "The ML model estimates churn probability. Retention suggestions come from "
    "separate business rules and are not causal conclusions."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Train it first: `python -m src.train`")
    st.stop()

pipeline = load_model()

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Customer details")
sb = st.sidebar
gender = sb.selectbox("Gender", ["Female", "Male"])
senior = sb.selectbox("Senior Citizen", ["No", "Yes"])
partner = sb.selectbox("Partner", ["No", "Yes"])
dependents = sb.selectbox("Dependents", ["No", "Yes"])
tenure = sb.slider("Tenure (months)", 0, 72, 12)

phone = sb.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = sb.selectbox(
    "Multiple Lines", ["No phone service"] if phone == "No" else ["No", "Yes"])
internet = sb.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
addon_options = ["No internet service"] if internet == "No" else ["No", "Yes"]
online_security = sb.selectbox("Online Security", addon_options)
online_backup = sb.selectbox("Online Backup", addon_options)
device_protection = sb.selectbox("Device Protection", addon_options)
tech_support = sb.selectbox("Tech Support", addon_options)
streaming_tv = sb.selectbox("Streaming TV", addon_options)
streaming_movies = sb.selectbox("Streaming Movies", addon_options)

contract = sb.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = sb.selectbox("Paperless Billing", ["Yes", "No"])
payment = sb.selectbox("Payment Method", [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = sb.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
total = sb.number_input(
    "Total Charges ($)", 0.0, 20000.0, float(round(tenure * monthly, 2)), step=1.0,
    key=f"total_{tenure}_{monthly}",
    help="Defaults to tenure x monthly charges; edit if you know the real value.")

predict = sb.button("Predict Churn", type="primary")

customer = {
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner, "Dependents": dependents, "tenure": tenure,
    "PhoneService": phone, "MultipleLines": multiple_lines,
    "InternetService": internet, "OnlineSecurity": online_security,
    "OnlineBackup": online_backup, "DeviceProtection": device_protection,
    "TechSupport": tech_support, "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": total,
}
# Column names/order must match the training schema.
X_new = pd.DataFrame([customer])[RAW_FEATURES]

# --------------------------------------------------------------- main panel
if not predict:
    st.info("Fill in the customer details in the sidebar and click **Predict Churn**.")
else:
    prob = float(pipeline.predict_proba(X_new)[0, 1])
    level = risk_level(prob)

    c1, c2 = st.columns(2)
    c1.metric("Churn Probability", f"{prob:.1%}")
    c2.metric("Risk Level", level)
    st.caption(
        f"Risk bands are project-defined: LOW < {LOW_THRESHOLD:.0%}, "
        f"HIGH > {HIGH_THRESHOLD:.0%}, otherwise MEDIUM. They are not universal standards."
    )
    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Prediction explanation")
        explainer = load_explainer(pipeline)
        if explainer is None:
            st.warning("SHAP is unavailable, so no explanation is shown.")
        else:
            up, down = explain.explain_customer(pipeline, explainer, X_new, top_n=5)
            st.markdown("**Factors increasing risk**")
            for name, val in up:
                st.write(f"• {name} (+{val:.3f})")
            st.markdown("**Factors reducing risk**")
            for name, val in down:
                st.write(f"• {name} ({val:.3f})")

            items = (up + down)
            if items:
                fig, ax = plt.subplots(figsize=(5, 3.2))
                items = sorted(items, key=lambda kv: kv[1])
                ax.barh([k for k, _ in items], [v for _, v in items],
                        color=["tab:blue" if v < 0 else "tab:red" for _, v in items])
                ax.axvline(0, color="black", lw=0.8)
                ax.set_xlabel("SHAP value (effect on model output)")
                fig.tight_layout()
                st.pyplot(fig)
            st.caption("SHAP describes how the model behaves, not what causes churn.")

    with right:
        st.subheader("Recommendation")
        for action in recommend(customer, prob):
            st.write(f"• {action}")
        st.caption("Rule-based suggestions, separate from the ML prediction.")

    with st.expander("Customer summary", expanded=True):
        st.dataframe(X_new.T.rename(columns={0: "Value"}).astype(str))

    with st.expander("Global feature importance (mean |SHAP| on a training sample)"):
        explainer = load_explainer(pipeline)
        if explainer is not None and SAMPLE_PATH.exists():
            imp = explain.global_importance(
                pipeline, explainer, pd.read_csv(SAMPLE_PATH)).head(12)
            st.bar_chart(imp.iloc[::-1])
        else:
            st.write("Not available.")
