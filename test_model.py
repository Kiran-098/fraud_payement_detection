import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")
print("\n✅ Loaded Model: fraud_detection_model.pkl")

# Load test dataset
df_test = pd.read_csv(r"C:\Users\kiran\OneDrive\Desktop\onlinefraud.csv")  # Replace with your actual test dataset

# Take only the first 10,000 rows
df_test = df_test.head(10000)

# Encode 'type' column using one-hot encoding
df_test = pd.get_dummies(df_test, columns=["type"], drop_first=True)

# Separate features and target
X_test = df_test.drop(columns=["isFraud"])
y_test = df_test["isFraud"]

# Ensure test data has same features as the model
model_features = model.get_booster().feature_names
missing_cols = set(model_features) - set(X_test.columns)
extra_cols = set(X_test.columns) - set(model_features)

# Add missing columns with zeros
for col in missing_cols:
    X_test[col] = 0

# Drop extra columns
X_test = X_test[model_features]

# Get fraud probabilities
fraud_probs = model.predict_proba(X_test)[:, 1]

# Apply the new fraud threshold of 0.61
y_pred = (fraud_probs >= 0.61).astype(int)

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, fraud_probs)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n📊 Model Evaluation:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1-Score: {f1:.4f}")
print(f"✅ ROC-AUC: {roc_auc:.4f}")
print(f"\n🔹 Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

