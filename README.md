# Bank Churn Prediction API

REST API serving the churn model from [bank-churn-retention-analysis](https://github.com/OkoroBlessing/bank-churn-rentention-analysis). Scores customer churn risk with a Random Forest at the cost-optimised threshold of 0.15, where the model catches 88% of churners.

## Example

Request `POST /predict`:

```json
{
  "credit_score": 600, "geography": "Germany", "gender": "Female",
  "age": 52, "tenure": 2, "balance": 120000, "num_of_products": 3,
  "has_credit_card": true, "is_active_member": false, "estimated_salary": 45000
}
```

Response:

```json
{
  "churn_probability": 0.944,
  "at_risk": true,
  "threshold": 0.15,
  "recommendation": "Offer retention incentive (budget GBP 40)"
}
```

Interactive docs auto-generate at `/docs` (Swagger UI).

## Files

- `train_model.py` — trains the Random Forest and exports `churn_model.pkl`
- `app.py` — FastAPI application: `/` health check, `/predict` scoring endpoint
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
# open http://localhost:8000/docs
```

## Deploy to Azure App Service (free tier)

Prerequisites: free Azure account (azure.microsoft.com/free), this repo pushed to GitHub with `churn_model.pkl` committed.

1. Portal: Create a resource → Web App.
2. Settings: Runtime **Python 3.11**, OS **Linux**, Pricing plan **Free F1**, Region **UK South**.
3. Deployment tab: enable **GitHub Actions**, sign in, pick this repo and branch. Azure writes the workflow file for you.
4. After creation: App Service → Configuration → General settings → Startup Command:

   ```
   gunicorn -w 2 -k uvicorn.workers.UvicornWorker app:app
   ```

5. Save, restart, then open `https://<your-app-name>.azurewebsites.net/docs`.

Every push to the branch redeploys automatically through the generated GitHub Action.

## Why threshold 0.15, not 0.50

A missed churner costs ~£200 (customer replacement) while a wasted retention offer costs £40. Sweeping the threshold against this cost matrix showed 0.15 minimises net cost, lifting churner recall from 58% to 88%. Full analysis in the [main project repo](https://github.com/OkoroBlessing/bank-churn-rentention-analysis).

## Limitations

- Model trained on 10,000 records from one bank; retrain before any real use.
- No authentication on the endpoint; add an API key before production traffic.
- Free tier sleeps after idle periods; the first request after sleep takes ~30 seconds.
