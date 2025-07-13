import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model, scaler, and the preprocessing pipeline
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')
preprocessor = joblib.load('preprocessor.pkl')  # assumed pipeline to get V1-V28

st.title("💳 Credit Card Fraud Detection (Raw Features)")

# Input fields for raw data
cc_num = st.text_input("Credit Card Number")
merchant = st.text_input("Merchant Name")
category = st.selectbox("Category", ["Food", "Travel", "Entertainment", "Other"])
gender = st.selectbox("Cardholder Gender", ["M", "F"])
lat = st.number_input("Latitude", value=0.0)
long = st.number_input("Longitude", value=0.0)
time = st.number_input("Transaction Time", value=0.0)
amount = st.number_input("Transaction Amount", value=0.0)

# Prepare input data as a DataFrame with correct columns for preprocessing
input_dict = {
    "cc_num": [cc_num],
    "merchant": [merchant],
    "category": [category],
    "gender": [gender],
    "lat": [lat],
    "long": [long],
    "time": [time],
    "amount": [amount]
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict"):
    # Preprocess input_df to the features your model expects (e.g., V1-V28 + time + amount)
    # This requires you have a saved preprocessing pipeline (preprocessor)
    processed = preprocessor.transform(input_df)
    
    # Scale processed input
    input_scaled = scaler.transform(processed)
    
    prediction = model.predict(input_scaled)[0]
    result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    st.subheader(f"Prediction: {result}")
