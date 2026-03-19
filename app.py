import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# PAGE TITLE
# -----------------------------
st.title("Customer Churn Prediction")

st.markdown("""
### Machine Learning Model
This application predicts whether a telecom customer is likely to **churn**
based on customer tenure and billing information.

**Model Used:** XGBoost Classifier
""")

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# USER INPUTS
# -----------------------------
st.subheader("Enter Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

with col2:
    monthly_charges = st.number_input("Monthly Charges", 0, 200, 70)

total_charges = st.number_input("Total Charges", 0, 10000, 1000)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Predict Churn"):

    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    df = pd.DataFrame([input_data])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]

    # Apply scaling
    df = scaler.transform(df)

    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ Customer likely to churn ({probability*100:.2f}% probability)")
    else:
        st.success(f"✅ Customer likely to stay ({(1-probability)*100:.2f}% probability)")