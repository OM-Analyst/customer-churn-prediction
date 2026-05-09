# =========================
# 02_preprocessing.py
# =========================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================
# 1. Load Dataset
# =========================

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


# =========================
# 2. Basic Cleaning
# =========================

# Drop customer ID because it is not useful for prediction
df = df.drop("customerID", axis=1)

# Convert TotalCharges from object/text to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing TotalCharges values with median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())


# =========================
# 3. Feature Engineering
# =========================

# Create tenure groups
df["TenureGroup"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    include_lowest=True
)

# Create average monthly spend
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"]

# Replace infinity values caused by tenure = 0
df["AvgMonthlySpend"] = df["AvgMonthlySpend"].replace([np.inf, -np.inf], 0)

# Fill any remaining missing values
df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(0)


# =========================
# 4. Encode Target Variable
# =========================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# =========================
# 5. Separate Features and Target
# =========================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# =========================
# 6. One-Hot Encoding
# =========================

X = pd.get_dummies(X, drop_first=True)


# =========================
# 7. Train/Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 8. Scaling
# =========================

scaler = StandardScaler()

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]

X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])


# =========================
# 9. Check Final Shapes
# =========================

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nPreprocessing completed successfully.")
# Save processed datasets

X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)

y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Processed files saved successfully.")