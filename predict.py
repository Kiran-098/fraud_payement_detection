import joblib
import pandas as pd
import numpy as np

# Load the trained fraud detection model
model = joblib.load("fraud_detection_model.pkl")

# Get the feature names expected by the model
model_features = model.get_booster().feature_names

# Function to calculate missing features
def calculate_features(transaction):
    transaction["transaction_velocity"] = transaction["amount"] / transaction["step"] if transaction["step"] > 0 else transaction["amount"]
    transaction["amount_deviation"] = abs(transaction["oldbalanceOrg"] - transaction["newbalanceOrig"])
    transaction["balance_change_ratio"] = transaction["amount"] / transaction["oldbalanceOrg"] if transaction["oldbalanceOrg"] > 0 else 0
    return transaction

# Function to preprocess the input transaction
def preprocess_input(transaction):
    # Compute missing features
    transaction = calculate_features(transaction)

    # Convert dictionary to DataFrame
    df = pd.DataFrame([transaction])

    # One-hot encode 'type' column
    df = pd.get_dummies(df, columns=['type'])

    # Ensure all model features exist in df
    missing_cols = set(model_features) - set(df.columns)
    extra_cols = set(df.columns) - set(model_features)

    # Add missing columns with zero
    for col in missing_cols:
        df[col] = 0

    # Drop extra columns
    df = df[model_features]

    return df

# Function to predict fraud
def predict_fraud(transaction):
    X = preprocess_input(transaction)
    fraud_prob = model.predict_proba(X)[:, 1][0]  # Get fraud probability
    is_fraud = 1 if fraud_prob >= 0.31 else 0  # Apply threshold
    return {"fraud_prediction": is_fraud, "fraud_probability": round(fraud_prob, 4)}

