import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Preprocessing
# Target: Disease
target_col = "Disease"
# Features: Remove Disease and Outcome Variable (Outcome is typically derived from having a disease)
X = df.drop(columns=[target_col, "Outcome Variable"])
y = df[target_col]

# Encode Categorical Features
# Mappings aligned with app.py logic
# Fever, Cough, Fatigue, Difficulty Breathing: Yes/No -> 1/0
for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
    X[col] = X[col].map({"Yes": 1, "No": 0})

# Gender: Male/Female -> 1/0
X["Gender"] = X["Gender"].map({"Male": 1, "Female": 0})

# Blood Pressure: Low/Normal/High -> 0/1/2
X["Blood Pressure"] = X["Blood Pressure"].map({"Low": 0, "Normal": 1, "High": 2})

# Cholesterol Level: Low/Normal/High -> 0/1/2
X["Cholesterol Level"] = X["Cholesterol Level"].map({"Low": 0, "Normal": 1, "High": 2})

# Verify all columns are numeric
print("Feature types:")
print(X.dtypes)

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save LabelEncoder classes to decode predictions later
np.save("disease_classes.npy", le.classes_)
print(f"Number of classes: {len(le.classes_)}")

# Train on FULL dataset to ensure all classes are present
# (Dataset is too small to split reliably with 100+ classes)
print("Training Multi-class XGBoost Model on Full Data...")
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=len(le.classes_),
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1
)

model.fit(X, y_encoded)

# Evaluate on self (just for sanity check)
y_pred = model.predict(X)
acc = accuracy_score(y_encoded, y_pred)
print(f"Model Accuracy (Full Data): {acc * 100:.2f}%")

# Save Model
joblib.dump(model, "multiclass_model.pkl")
print("Model saved to multiclass_model.pkl")
