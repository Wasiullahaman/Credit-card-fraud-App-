import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Fraud Detector", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")

uploaded_file = st.file_uploader("📂 Upload a CSV File", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    file_details = {
        "Filename": uploaded_file.name,
        "Size (KB)": round(len(uploaded_file.getvalue()) / 1024, 2),
        "Rows": len(data)
    }

    st.sidebar.header("📁 File Info")
    st.sidebar.write(file_details)

    # Drop 'Class' column if exists
    if 'Class' in data.columns:
        data = data.drop(columns=['Class'])

    if 'Amount' in data.columns:
        data['Amount'] = scaler.transform(data[['Amount']])

    # Predict and get probabilities
    predictions = model.predict(data)
    probabilities = model.predict_proba(data)[:, 1]  # Prob of fraud class

    data['Prediction'] = predictions
    data['Fraud Probability'] = probabilities.round(3)

    fraud_data = data[data['Prediction'] == 1]
    normal_data = data[data['Prediction'] == 0]

    fraud_count = len(fraud_data)
    total = len(data)
    fraud_rate = round((fraud_count / total) * 100, 2)

    # Sidebar metrics
    st.sidebar.header("📊 Fraud Stats")
    st.sidebar.metric("Total Transactions", total)
    st.sidebar.metric("Fraudulent", fraud_count)
    st.sidebar.metric("Fraud Rate (%)", fraud_rate)

    # Main summary
    st.subheader("📋 Prediction Summary")
    st.success(f"✅ Total Transactions: {total}")
    st.warning(f"⚠️ Fraudulent Transactions: {fraud_count}")
    st.info(f"📈 Fraud Rate: {fraud_rate}%")

    # Filter toggle
    st.subheader("🔍 View Transactions")
    view_option = st.radio("Select what to view:", ["All", "Only Fraud", "Only Non-Fraud"])
    if view_option == "Only Fraud":
        st.dataframe(fraud_data)
    elif view_option == "Only Non-Fraud":
        st.dataframe(normal_data)
    else:
        st.dataframe(data.style.applymap(
            lambda val: 'background-color: #FFCCCC' if val == 1 else '',
            subset=['Prediction']
        ))

    # Charts
    st.subheader("📈 Fraud Breakdown")
    fig_pie = px.pie(
        names=['Non-Fraud', 'Fraud'],
        values=[len(normal_data), len(fraud_data)],
        title="Fraud vs Non-Fraud",
        color_discrete_map={'Non-Fraud': 'green', 'Fraud': 'red'},
        hole=0.4
    )
    st.plotly_chart(fig_pie)

    st.subheader("📊 Transaction Counts")
    count_df = pd.DataFrame({
        'Class': ['Non-Fraud', 'Fraud'],
        'Count': [len(normal_data), len(fraud_data)]
    })
    fig_bar = px.bar(
        count_df,
        x='Class',
        y='Count',
        color='Class',
        color_discrete_map={'Non-Fraud': 'green', 'Fraud': 'red'},
        title="Transaction Type Counts"
    )
    st.plotly_chart(fig_bar)

    # Download button
    st.subheader("⬇️ Download Results")
    st.download_button(
        "Download as CSV",
        data=data.to_csv(index=False).encode('utf-8'),
        file_name="fraud_detection_results.csv",
        mime="text/csv"
    )
