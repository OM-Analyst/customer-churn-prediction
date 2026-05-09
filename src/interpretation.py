# =========================
# 04_interpretation.py
# =========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from xgboost import XGBClassifier


# =========================
# 1. Load Processed Data
# =========================

X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")

y_train = pd.read_csv("outputs/y_train.csv").squeeze()
y_test = pd.read_csv("outputs/y_test.csv").squeeze()


# =========================
# 2. Train Final Model
# =========================

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)


# =========================
# 3. Feature Importance
# =========================

importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))


# =========================
# 4. Plot Feature Importance
# =========================

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.show()


# =========================
# 5. SHAP Interpretation
# =========================

explainer = shap.Explainer(model)

shap_values = explainer(X_test)


# =========================
# 6. SHAP Summary Plot
# =========================

shap.summary_plot(
    shap_values,
    X_test
)


# =========================
# 7. Business Recommendations
# =========================

print("\n==============================")
print("BUSINESS RECOMMENDATIONS")
print("==============================")

print("""
1. Customers on month-to-month contracts should be targeted
   with long-term contract incentives to reduce churn risk.

2. Customers with high monthly charges may benefit from
   loyalty discounts or bundled service packages.

3. New customers with low tenure should receive stronger
   onboarding and customer engagement support.

4. Customers identified as high churn risk should be targeted
   with retention campaigns before cancellation occurs.

5. Service quality improvements should focus on customer groups
   showing higher churn probabilities.
""")