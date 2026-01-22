import joblib
import numpy as np
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxconverter_common.data_types import FloatTensorType
import xgboost as xgb

# Load the trained binary model
print("Loading model...")
model = joblib.load("binary_model.pkl")

# Define input type
# 8 features: Age, Gender, Ethnicity, Fever, Cough, Fatigue, BP, Cholesterol
# Note: Ensure the feature count matches exactly what was used in training.
# In train_binary.py, we dropped 'Disease' and 'Outcome Variable'.
# Columns used: Fever, Cough, Fatigue, Difficulty Breathing, Age, Gender, Blood Pressure, Cholesterol Level
# That's 8 columns.
initial_type = [('float_input', FloatTensorType([None, 8]))]

# Convert to ONNX
print("Converting to ONNX...")
onnx_model = convert_xgboost(model, initial_types=initial_type)

# Save
with open("binary_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Model saved to binary_model.onnx")
