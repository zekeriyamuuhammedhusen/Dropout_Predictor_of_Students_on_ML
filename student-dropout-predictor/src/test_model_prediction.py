import joblib
import pandas as pd
import os
from data_processing import load_data, preprocess_data, get_project_root

# Load model
project_root = get_project_root()
model_path = os.path.join(project_root, 'model', 'logistic_regression_model.pkl')
model = joblib.load(model_path)

# Load and preprocess data
df = load_data()
df = preprocess_data(df)

# Use the first row as a test
X = df.drop('Dropped_Out', axis=1)
sample = X.iloc[0:1]

# Predict
prob = model.predict_proba(sample)[0]
print('Probability not dropping out:', prob[0])
print('Probability dropping out:', prob[1])
