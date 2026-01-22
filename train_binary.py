import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Preprocessing
# Target: Outcome Variable (Positive/Negative)
target_col = "Outcome Variable"
# Features: Remove Disease (we are predicting outcome, not specific disease) and Target
X = df.drop(columns=["Disease", target_col])
y = df[target_col]

# Encode Categorical Features (Same as before)
for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
    X[col] = X[col].map({"Yes": 1, "No": 0})

X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})
X["Blood Pressure"] = X["Blood Pressure"].map({"Low": 0, "Normal": 1, "High": 2})
X["Cholesterol Level"] = X["Cholesterol Level"].map({"Low": 0, "Normal": 1, "High": 2})

# Encode Target: Positive -> 1, Negative -> 0
y_encoded = y.map({"Positive": 1, "Negative": 0})

print("Training Binary XGBoost Model...")
# Use the full dataset for production
model = xgb.XGBClassifier(
    objective="binary:logistic",
    use_label_encoder=False,
    eval_metric="logloss",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X, y_encoded)

# Evaluate on self
y_pred = model.predict(X)
acc = accuracy_score(y_encoded, y_pred)
print(f"Model Accuracy (Full Data): {acc * 100:.2f}%")

# Save Model
joblib.dump(model, "binary_model.pkl")
print("Model saved to binary_model.pkl")
