"""Train the final churn model and export it for the API.
Run once: python train_model.py -> creates churn_model.pkl"""
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

URL = 'https://raw.githubusercontent.com/OkoroBlessing/Datascience-dataset/master/Churn_Modelling.csv'
THRESHOLD = 0.15  # cost-optimal threshold from the analysis

df = pd.read_csv(URL).dropna().drop_duplicates()
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
df['Gender'] = (df['Gender'] == 'Male').astype(int)
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

X = df.drop('Exited', axis=1)
y = df['Exited'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
model = RandomForestClassifier(
    n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
model.fit(X_sm, y_sm)

joblib.dump({'model': model, 'features': list(X.columns), 'threshold': THRESHOLD},
            'churn_model.pkl')
print('Saved churn_model.pkl | features:', list(X.columns))
