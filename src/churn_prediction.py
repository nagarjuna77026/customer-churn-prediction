import pickle
import pandas as pd

print("Script started...")

# load trained model
model = pickle.load(open("models/churn_model.pkl", "rb"))

# load training feature columns
model_columns = pickle.load(open("models/model_columns.pkl", "rb"))

# load scaler
scaler = pickle.load(open("models/scaler.pkl", "rb"))


def predict_churn(input_data):

    df = pd.DataFrame([input_data])

    # create all required columns
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # reorder columns to match training data
    df = df[model_columns]

    # apply scaling
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)
    probability = model.predict_proba(df_scaled)[0][1]

    if prediction[0] == 1:
        return f"Customer will churn (Probability: {probability*100:.2f}%)"
    else:
        return f"Customer will stay (Probability: {(1-probability)*100:.2f}%)"


customer = {
    "tenure": 12,
    "MonthlyCharges": 80,
    "TotalCharges": 960
}

result = predict_churn(customer)

print("Prediction:", result)