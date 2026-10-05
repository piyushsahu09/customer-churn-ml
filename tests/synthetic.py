"""Synthetic data with the IBM Telco schema, used ONLY for tests/smoke runs."""
import numpy as np
import pandas as pd


def make_synthetic_telco(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yn = lambda p=0.5: rng.choice(["Yes", "No"], n, p=[p, 1 - p])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    phone = yn(0.9)
    def addon():
        a = yn(0.4)
        return np.where(internet == "No", "No internet service", a)
    tenure = rng.integers(0, 73, n)
    monthly = np.round(rng.uniform(18, 118, n), 2)
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yn(), "Dependents": yn(0.3), "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", yn()),
        "InternetService": internet,
        "OnlineSecurity": addon(), "OnlineBackup": addon(),
        "DeviceProtection": addon(), "TechSupport": addon(),
        "StreamingTV": addon(), "StreamingMovies": addon(),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
        "PaperlessBilling": yn(0.6),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check",
                                     "Bank transfer (automatic)", "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
    })
    total = np.round(df["tenure"] * df["MonthlyCharges"], 2).astype(str)
    total[df["tenure"] == 0] = " "          # same quirk as the real file
    df["TotalCharges"] = total
    logit = (-1.0 + 1.4 * (df["Contract"] == "Month-to-month") - 0.03 * df["tenure"]
             + 0.012 * df["MonthlyCharges"])
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")
    return df
