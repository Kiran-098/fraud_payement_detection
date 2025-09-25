import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler

print("\n🔹 Loading dataset...\n")

# Load dataset
df = pd.read_csv(r"C:\Users\kiran\OneDrive\Desktop\onlinefraud.csv")

# Drop unnecessary columns
df = df.drop(["nameOrig", "nameDest"], axis=1)

# Convert categorical feature "type" to numeric
df = pd.get_dummies(df, columns=["type"], drop_first=True)

# Feature Engineering
df["transaction_velocity"] = df.groupby("step")["amount"].transform("count")
df["amount_deviation"] = abs(df["amount"] - df.groupby("step")["amount"].transform("mean"))
df["balance_change_ratio"] = (df["oldbalanceOrg"] - df["newbalanceOrig"]) / (df["oldbalanceOrg"] + 1e-9)

# Splitting features and target
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply Borderline-SMOTE for oversampling fraud cases
smote = BorderlineSMOTE(sampling_strategy=0.3, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Apply Random Undersampling to balance normal cases
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_resampled, y_train_resampled = undersample.fit_resample(X_train_resampled, y_train_resampled)

# Show class distribution after resampling
print("\n🔹 Class distribution in Training Set after Resampling:")
print(y_train_resampled.value_counts(normalize=True).rename("proportion"))

print("\n🔹 Training XGBoost model...\n")

# Adjusted XGBoost parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=300,       # Reduce number of trees (to prevent overfitting)
    learning_rate=0.1,
    max_depth=4,            # Lower tree depth
    scale_pos_weight=10,    # Increase weight of fraud cases
    objective='binary:logistic',
    eval_metric="logloss",
    random_state=42
)

# Train the model
xgb_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]  # Get probability of fraud
threshold = 0.5  # Adjust threshold if needed
y_pred = (y_pred_prob >= threshold).astype(int)

# Model Evaluation
print("\n🔹 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n🔹 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n🔹 ROC-AUC Score:", roc_auc_score(y_test, y_pred_prob))

# Checking fraud detection effectiveness
print("\n🔹 Sample Fraud Predictions (Threshold = 0.5):")
fraud_cases = X_test[y_test == 1].iloc[:10]  # Select first 10 fraud cases
fraud_probs = xgb_model.predict_proba(fraud_cases)[:, 1]  # Fraud probability

for i, prob in enumerate(fraud_probs):
    print(f"Transaction {i+1}: Fraud Probability = {prob:.2f}, Predicted Fraud = {int(prob >= threshold)}")

# Save the trained model
joblib.dump(xgb_model, "fraud_detection_model.pkl")
print("\n✅ Model saved as fraud_detection_model.pkl")
