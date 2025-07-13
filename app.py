import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Drop 'Class' column if it exists
    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])

    if 'Amount' in data.columns:
        data['Amount'] = scaler.transform(data[['Amount']])

    predictions = model.predict(data)
    data['Prediction'] = predictions

    st.write("🔍 Prediction Results:", data.head())

    fraud_data = data[data['Prediction'] == 1]
    st.warning(f"⚠️ Fraudulent Transactions Detected: {len(fraud_data)}")
