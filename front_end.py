"""
Churn Risk Scorer — human-facing front-end for the churn model.
The REST API (app.py) serves other systems; this serves people.
Both load the same churn_model.pkl, so scores always agree.
Run: streamlit run frontend.py
Author: Blessing Okoro
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Risk Scorer", page_icon="🏦", layout="centered")
ACCENT = "#1F4E5F"

@st.cache_resource
def load_model():
    artifact = joblib.load("churn_model.pkl")
    return artifact["model"], artifact["features"], artifact["threshold"]

MODEL, FEATURES, THRESHOLD = load_model()

st.title("Churn Risk Scorer")
st.markdown(
    "Score a bank customer's churn risk with the Random Forest from the "
    "[churn analysis](https://github.com/OkoroBlessing/bank-churn-rentention-analysis). "
    f"The decision threshold is **{THRESHOLD}**, not 0.50 — a missed churner costs "
    "roughly 5× a wasted retention offer, so the model deliberately casts a wide net "
    "and catches 88% of churners.")

with st.form("customer"):
    st.subheader("Customer profile")
    c1, c2 = st.columns(2)
    with c1:
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.slider("Age", 18, 95, 40)
        tenure = st.slider("Tenure with bank (years)", 0, 10, 3)
        num_products = st.selectbox("Number of products", [1, 2, 3, 4])
    with c2:
        credit_score = st.slider("Credit score", 300, 900, 650)
        balance = st.number_input("Account balance (£)", 0.0, 500_000.0, 75_000.0, step=1_000.0)
        salary = st.number_input("Estimated salary (£)", 0.0, 500_000.0, 50_000.0, step=1_000.0)
        has_card = st.toggle("Has credit card", value=True)
        active = st.toggle("Active member", value=True)
    submitted = st.form_submit_button("Score customer", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": 1 if gender == "Male" else 0,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_card),
        "IsActiveMember": int(active),
        "EstimatedSalary": salary,
        "Geography_Germany": 1 if geography == "Germany" else 0,
        "Geography_Spain": 1 if geography == "Spain" else 0,
    }])[FEATURES]

    proba = float(MODEL.predict_proba(row)[0, 1])
    at_risk = proba >= THRESHOLD

    st.divider()
    a, b = st.columns(2)
    a.metric("Churn probability", f"{proba:.1%}")
    b.metric("Decision threshold", f"{THRESHOLD:.0%}")
    st.progress(min(proba, 1.0))

    if at_risk:
        st.error("**AT RISK** — recommend a retention offer (budget £40 per customer).")
    else:
        st.success("**Low risk** — no action needed.")

    with st.expander("Why this recommendation"):
        st.markdown(
            f"- The model estimates a **{proba:.1%}** probability this customer "
            f"leaves.\n"
            f"- Any score at or above **{THRESHOLD:.0%}** triggers the offer, because "
            "cost analysis showed a missed churner (~£200) costs far more than a "
            "wasted offer (£40). The threshold minimises total cost, not error count.\n"
            "- Top churn drivers in this model: age, account inactivity, balance, "
            "and the German market.")

st.divider()
st.caption(
    "Same model artefact as the REST API, so scores agree across both surfaces. "
    "Trained on 10,000 records from one bank — a demonstration, not a production "
    "system. Blessing Okoro · MSc Artificial Intelligence (Distinction)")
