import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Drop 'Class' column if it exists
    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])

    # Scale 'Amount' column
    if 'Amount' in data.columns:
        data['Amount'] = scaler.transform(data[['Amount']])

    # Make predictions
    predictions = model.predict(data)
    data['Prediction'] = predictions

    # Summary info
    fraud_data = data[data['Prediction'] == 1]
    normal_data = data[data['Prediction'] == 0]
    fraud_count = len(fraud_data)
    total_count = len(data)
    fraud_rate = round((fraud_count / total_count) * 100, 2)

    # Output summary
    st.subheader("🔍 Prediction Summary")
    st.success(f"✅ Total Transactions: {total_count}")
    st.warning(f"⚠️ Fraudulent Transactions Detected: {fraud_count}")
    st.info(f"📊 Fraud Rate: {fraud_rate}%")

    st.subheader("📋 Preview of Results")
    st.dataframe(data.head())

    # Pie chart
    st.subheader("📈 Fraud vs Non-Fraud Pie Chart")
    pie_fig = px.pie(data, names='Prediction', title="Fraud Detection Breakdown",
                     color_discrete_map={0: 'green', 1: 'red'},
                     hole=0.4)
    st.plotly_chart(pie_fig)

    # Bar chart
      st.subheader("📊 Count of Transactions by Type")

    # Create a new DataFrame for plotting
    counts = data['Prediction'].value_counts().sort_index()
    count_df = pd.DataFrame({
        'Class': ['Non-Fraud', 'Fraud'],
        'Count': [counts.get(0, 0), counts.get(1, 0)]
    })

    bar_fig = px.bar(
        count_df,
        x='Class',
        y='Count',
        color='Class',
        color_discrete_map={'Non-Fraud': 'green', 'Fraud': 'red'},
        title="Number of Fraud vs Non-Fraud Transactions"
    )
    st.plotly_chart(bar_fig)


    # Download results
    st.subheader("⬇️ Download Prediction Results")
    st.download_button(
        label="Download as CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )
