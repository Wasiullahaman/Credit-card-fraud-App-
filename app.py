import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="💳 Fraud Detection Dashboard", layout="wide")
st.title("🚨 Credit Card Fraud Detection System")

uploaded_file = st.file_uploader("📂 Upload transaction CSV file", type=["csv"])

if uploaded_file:
    # Load and preview file
    df = pd.read_csv(uploaded_file)
    st.sidebar.header("📁 File Info")
    st.sidebar.write({
        "Filename": uploaded_file.name,
        "Size (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
        "Rows": len(df)
    })

    # Drop class column if exists
    if 'Class' in df.columns:
        df.drop(columns=['Class'], inplace=True)

    # Scale 'Amount' column
    if 'Amount' in df.columns:
        df['Amount'] = scaler.transform(df[['Amount']])

    # Predict
    y_pred = model.predict(df)
    y_proba = model.predict_proba(df)[:, 1]  # Probability of class 1

    # Add prediction + scoring
    df['Fraud_Prediction'] = y_pred
    df['Fraud_Score'] = y_proba.round(3)

    # Action Logic
    def risk_action(score):
        if score > 0.85:
            return "🔴 Block"
        elif score > 0.6:
            return "🟠 Monitor"
        else:
            return "🟢 Safe"

    df['Action'] = df['Fraud_Score'].apply(risk_action)

    # Summary
    total = len(df)
    frauds = df[df['Fraud_Prediction'] == 1]
    fraud_count = len(frauds)
    fraud_rate = round((fraud_count / total) * 100, 2)

    st.sidebar.header("📊 Fraud Stats")
    st.sidebar.metric("Total", total)
    st.sidebar.metric("Fraudulent", fraud_count)
    st.sidebar.metric("Fraud Rate (%)", fraud_rate)

    # Filter View
    st.subheader("🔍 Filter Results")
    view_option = st.radio("View:", ["All", "Only Fraud", "Only Non-Fraud"], horizontal=True)
    if view_option == "Only Fraud":
        st.dataframe(df[df['Fraud_Prediction'] == 1])
    elif view_option == "Only Non-Fraud":
        st.dataframe(df[df['Fraud_Prediction'] == 0])
    else:
        st.dataframe(df)

    # Charts
    st.subheader("📊 Fraud Distribution")
    pie_fig = px.pie(
        names=["Non-Fraud", "Fraud"],
        values=[total - fraud_count, fraud_count],
        color_discrete_sequence=["green", "red"],
        title="Fraud vs Non-Fraud Transactions"
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    st.subheader("💰 Fraud Amounts")
    fraud_by_amount = frauds.copy()
    fraud_by_amount['Transaction_ID'] = fraud_by_amount.index
    line_fig = px.scatter(
        fraud_by_amount, x='Transaction_ID', y='Amount',
        color='Fraud_Score',
        title="Fraud Amounts & Risk Scores",
        color_continuous_scale='reds'
    )
    st.plotly_chart(line_fig, use_container_width=True)

    # Download
    st.subheader("⬇️ Download Results")
    st.download_button(
        "Download as CSV",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )

    # Placeholder for SHAP (optional)
    st.subheader("🧠 Coming Soon: Explainability with SHAP")
    st.info("SHAP (SHapley Additive exPlanations) helps explain *why* a transaction was marked as fraud.")
