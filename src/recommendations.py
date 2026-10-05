"""Risk segmentation and rule-based retention suggestions.

IMPORTANT: this layer is BUSINESS RULES applied on top of the ML prediction.
The model predicts risk; it does not know which action would prevent churn.
These rules are heuristics to validate with experiments (e.g. A/B tests),
not causal conclusions.
"""
from __future__ import annotations

# Project-defined operational thresholds (NOT universal standards). Tune them
# to the retention team's capacity and the cost of contacting a customer.
LOW_THRESHOLD = 0.30
HIGH_THRESHOLD = 0.70

# Approximate "high bill" cut-off. Check it against your own EDA
# (e.g. the median or 75th percentile of MonthlyCharges) and adjust.
HIGH_MONTHLY_CHARGES = 70.0


def risk_level(prob: float, low: float = LOW_THRESHOLD, high: float = HIGH_THRESHOLD) -> str:
    """LOW if prob < low, HIGH if prob > high, otherwise MEDIUM."""
    if prob < low:
        return "LOW"
    if prob <= high:
        return "MEDIUM"
    return "HIGH"


def recommend(customer: dict, prob: float, low: float = LOW_THRESHOLD,
              high: float = HIGH_THRESHOLD) -> list[str]:
    """Return suggested actions for a customer (dict using the raw schema)."""
    level = risk_level(prob, low, high)
    if level == "LOW":
        return ["No retention action needed now. Keep monitoring."]

    actions = []
    has_internet = customer.get("InternetService") != "No"

    if customer.get("Contract") == "Month-to-month":
        actions.append("Consider offering a longer-term contract incentive.")
    if float(customer.get("MonthlyCharges", 0)) >= HIGH_MONTHLY_CHARGES:
        actions.append("Consider reviewing pricing / plan suitability with the customer.")
    if has_internet and customer.get("TechSupport") == "No":
        actions.append("Prioritize a customer-support follow-up; consider offering tech support.")
    if has_internet and customer.get("OnlineSecurity") == "No":
        actions.append("Consider offering a trial of online security services.")
    if customer.get("PaymentMethod") == "Electronic check":
        actions.append("Suggest automatic payment methods (card / bank transfer).")
    if int(customer.get("tenure", 99)) <= 12:
        actions.append("Add the customer to an early-life onboarding / check-in programme.")

    if not actions:
        actions.append("Reach out for general satisfaction feedback.")
    if level == "HIGH":
        actions.insert(0, "Flag for priority outreach by the retention team.")
    return actions
