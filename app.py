import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    layout="wide"
)

# Load files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
metrics = joblib.load("model_metrics.pkl")
df = pd.read_csv("cleaned_churn_data.csv")

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend"]

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "EDA Dashboard",
        "Model Performance",
        "Feature Importance",
        "Predict Churn",
        "Business Recommendations"
    ]
)

# Home
if page == "Home":
    st.title("Customer Churn Prediction Dashboard")

    st.write("""
    This dashboard analyzes telecom customer churn and predicts whether a customer is likely to leave.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Customers", len(df))
    col2.metric("Churned Customers", df[df["Churn"] == 1].shape[0])
    col3.metric("Stayed Customers", df[df["Churn"] == 0].shape[0])

    churn_rate = df["Churn"].mean() * 100
    st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")

# EDA
elif page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")

    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Churn", ax=ax)
    ax.set_xticklabels(["No", "Yes"])
    st.pyplot(fig)

    st.subheader("Monthly Charges by Churn")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
    ax.set_xticklabels(["No", "Yes"])
    st.pyplot(fig)

    st.subheader("Tenure by Churn")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", kde=True, bins=30, ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Internet Service")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax)
    st.pyplot(fig)

    st.subheader("Churn by Payment Method")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=ax)
    plt.xticks(rotation=35)
    st.pyplot(fig)

# Model Performance
elif page == "Model Performance":
    st.title("Model Performance")

    st.write("Final model used: **XGBoost Classifier**")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Accuracy", f"{metrics['Accuracy']:.2f}")
    col2.metric("Precision", f"{metrics['Precision']:.2f}")
    col3.metric("Recall", f"{metrics['Recall']:.2f}")
    col4.metric("F1 Score", f"{metrics['F1 Score']:.2f}")
    col5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.2f}")

    st.write("""
    The model performs well overall.  
    ROC-AUC shows how well the model separates churners from non-churners.
    """)

# Feature Importance
elif page == "Feature Importance":
    st.title("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importance_df.head(15))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=importance_df.head(10),
        x="Importance",
        y="Feature",
        ax=ax
    )
    ax.set_title("Top 10 Most Important Features")
    st.pyplot(fig)

    st.write("""
    The most important churn drivers include contract type, internet service type,
    tenure, payment method, and monthly spending.
    """)

# Predict Churn
elif page == "Predict Churn":
    st.title("Predict Customer Churn")

    st.write("Enter customer details below:")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    tenure = st.slider("Tenure Months", 0, 72, 12)

    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=800.0)

    if tenure == 0:
        avg_monthly_spend = 0
    else:
        avg_monthly_spend = total_charges / tenure

    tenure_group = pd.cut(
        [tenure],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
        include_lowest=True
    )[0]

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "TenureGroup": [tenure_group],
        "AvgMonthlySpend": [avg_monthly_spend]
    })

    input_encoded = pd.get_dummies(input_data, drop_first=True)

    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

    if st.button("Predict Churn"):
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"High Churn Risk: {probability:.2%}")
        else:
            st.success(f"Low Churn Risk: {probability:.2%}")

        st.write("Churn probability means how likely the customer is to leave.")

# Recommendations
elif page == "Business Recommendations":
    st.title("Business Recommendations")

    st.write("""
    Based on the model and EDA results, the main recommendations are:
    """)

    st.markdown("""
    ### 1. Target month-to-month customers
    Customers on month-to-month contracts are more likely to churn.  
    Offer discounts or benefits to encourage one-year or two-year contracts.

    ### 2. Support new customers early
    Customers with low tenure have higher churn risk.  
    Improve onboarding, welcome offers, and early customer support.

    ### 3. Review high monthly charges
    Customers with higher monthly charges are more likely to churn.  
    Offer loyalty discounts or bundle packages.

    ### 4. Investigate fiber optic churn
    Fiber optic customers showed high churn influence.  
    The company should review pricing, service quality, and customer satisfaction.

    ### 5. Use the model for retention campaigns
    The model can identify high-risk customers before they leave.  
    These customers can be targeted with retention offers.
    """)