# 1. Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# 2. Load Dataset

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# View first rows
print(df.head())

# 3. Basic Dataset Overview

print("Dataset shape:", df.shape)

df.info()

df.describe()

# 4. Check Missing Values
print(df.isnull().sum())

# 5. Check Duplicate Rows
print("Duplicate rows:", df.duplicated().sum())

# 6. Convert TotalCharges to Numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print(df["TotalCharges"].isnull().sum())

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# 7. Churn Distribution

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)

# 8. Churn by Gender

plt.figure(figsize=(6,4))
sns.countplot(data=df, x="gender", hue="Churn")
plt.title("Churn by Gender")
plt.show()

# 9. Churn by Senior Citizen
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="SeniorCitizen", hue="Churn")
plt.title("Churn by Senior Citizen")
plt.xlabel("Senior Citizen: 0 = No, 1 = Yes")
plt.show()

# 10. Churn by Contract Type

plt.figure(figsize=(8,4))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.show()

# 16. Numerical Correlation Heatmap

numeric_df = df.select_dtypes(include=["int64", "float64"])

plt.figure(figsize=(8,5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 13. Tenure Distribution by Churn

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="tenure", hue="Churn", kde=True, bins=30)
plt.title("Tenure Distribution by Churn")
plt.show()

# 14. Monthly Charges by Churn

plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges by Churn")
plt.show()

# =========================
# INSIGHTS FROM EDA
# =========================

"""
1. Customers on month-to-month contracts have the highest churn rate,
   indicating low commitment increases churn risk.

2. Customers with low tenure are more likely to churn,
   suggesting early-stage retention strategies are critical.

3. Higher monthly charges are associated with increased churn,
   indicating price sensitivity among customers.

"""