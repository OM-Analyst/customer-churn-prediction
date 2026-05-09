import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

# Load data
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean data
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Feature engineering
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    include_lowest=True
)

df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]
df["AvgMonthlySpend"] = df["AvgMonthlySpend"].replace([np.inf, -np.inf], 0)
df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(0)

# Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Split features/target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# One-hot encode
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numeric columns
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]

scaler = StandardScaler()

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Train model
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

print(metrics)

# Save files for Streamlit
joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")
joblib.dump(metrics, "models/model_metrics.pkl")

df.to_csv("data/cleaned_churn_data.csv", index=False)

print("Model and files saved successfully.")