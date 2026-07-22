"""Churn prediction REST API.
Serves the Random Forest model at the cost-optimal threshold (0.15).
Run locally: uvicorn app:app --reload
Docs: http://localhost:8000/docs
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

artifact = joblib.load('churn_model.pkl')
MODEL = artifact['model']
FEATURES = artifact['features']
THRESHOLD = artifact['threshold']

app = FastAPI(
    title='Bank Churn Prediction API',
    description='Scores customer churn risk using a Random Forest at a '
                'cost-optimised decision threshold (0.15). A missed churner '
                'costs 5x a wasted retention offer, so the threshold favours recall.',
    version='1.0.0')


class Customer(BaseModel):
    credit_score: int = Field(..., ge=300, le=900, examples=[650])
    geography: str = Field(..., examples=['France'], description='France, Germany or Spain')
    gender: str = Field(..., examples=['Female'], description='Male or Female')
    age: int = Field(..., ge=18, le=100, examples=[42])
    tenure: int = Field(..., ge=0, le=10, examples=[3])
    balance: float = Field(..., ge=0, examples=[75000.0])
    num_of_products: int = Field(..., ge=1, le=4, examples=[1])
    has_credit_card: bool = Field(..., examples=[True])
    is_active_member: bool = Field(..., examples=[False])
    estimated_salary: float = Field(..., ge=0, examples=[50000.0])


def to_features(c: Customer) -> pd.DataFrame:
    row = {
        'CreditScore': c.credit_score,
        'Gender': 1 if c.gender.lower() == 'male' else 0,
        'Age': c.age,
        'Tenure': c.tenure,
        'Balance': c.balance,
        'NumOfProducts': c.num_of_products,
        'HasCrCard': int(c.has_credit_card),
        'IsActiveMember': int(c.is_active_member),
        'EstimatedSalary': c.estimated_salary,
        'Geography_Germany': 1 if c.geography.lower() == 'germany' else 0,
        'Geography_Spain': 1 if c.geography.lower() == 'spain' else 0,
    }
    return pd.DataFrame([row])[FEATURES]


@app.get('/')
def health():
    return {'status': 'ok', 'model': 'RandomForest', 'threshold': THRESHOLD}


@app.post('/predict')
def predict(customer: Customer):
    proba = float(MODEL.predict_proba(to_features(customer))[0, 1])
    at_risk = proba >= THRESHOLD
    return {
        'churn_probability': round(proba, 3),
        'at_risk': at_risk,
        'threshold': THRESHOLD,
        'recommendation': ('Offer retention incentive (budget GBP 40)'
                           if at_risk else 'No action needed'),
    }
