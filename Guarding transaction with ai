# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create a synthetic dataset (for demonstration)
# Features: Time (seconds since first transaction), Amount, V1-V4 (anonymized features), Class (0: Non-Fraud, 1: Fraud)
data = {
    'Time': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Amount': [100.0, 50.0, 2000.0, 75.0, 150.0, 3000.0, 25.0, 500.0, 10000.0, 80.0],
    'V1': [-1.359, 1.191, -5.0, 0.966, -0.185, -7.0, 1.792, -0.418, -10.0, 1.257],
    'V2': [0.072, -0.173, 4.0, -0.287, 0.669, 5.5, -0.863, 0.403, 7.0, -0.211],
    'V3': [2.536, 0.405, -6.0, 1.798, 1.974, -8.0, 0.095, 0.762, -12.0, 0.988],
    'V4': [1.378, -0.338, 3.5, -0.094, 0.456, 4.0, -0.631, 0.175, 6.0, -0.403],
    'Class': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]  # 0: Non-Fraud, 1: Fraud
}
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Scale Time and Amount
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Step 3: Train a Random Forest Classifier
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("\nModel Performance on Test Set:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
joblib.dump(rf_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Step 4: Simulate Real-Time Transaction Processing
# Example new transactions (mimicking incoming data)
new_transactions = pd.DataFrame({
    'Time': [10, 11, 12],
    'Amount': [120.0, 4500.0, 60.0],
    'V1': [1.0, -8.0, 0.8],
    'V2': [-0.2, 6.0, -0.3],
    'V3': [1.5, -10.0, 1.2],
    'V4': [-0.5, 5.0, -0.1]
})

# Load the model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Scale the new transactions
new_transactions[['Time', 'Amount']] = scaler.transform(new_transactions[['Time', 'Amount']])

# Predict fraud
predictions = model.predict(new_transactions)
probabilities = model.predict_proba(new_transactions)[:, 1]  # Probability of fraud

# Step 5: Output Results
print("\nNew Transaction Analysis:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    transaction = new_transactions.iloc[i]
    status = "Fraud" if pred == 1 else "Non-Fraud"
    print(f"\nTransaction {i+1}:")
    print(f"Time: {transaction['Time']:.2f}, Amount: ${transaction['Amount']:.2f}")
    print(f"Status: {status}, Fraud Probability: {prob:.2%}")
    if pred == 1:
        print("Action: Flag for review - High-risk transaction detected!")

# Step 6: Explain Flagged Transactions (Feature Importance for Fraud Cases)
if any(predictions == 1):
    print("\nFeature Importance for Fraud Detection:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)
