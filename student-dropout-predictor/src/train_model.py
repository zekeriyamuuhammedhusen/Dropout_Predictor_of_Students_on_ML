from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from data_processing import load_data, preprocess_data, get_project_root
import os

# Load the processed dataframe
df = load_data()
df = preprocess_data(df)

# Prepare the features and target variable
X = df.drop('Dropped_Out', axis=1)
y = df['Dropped_Out']

print(X.columns)

# Train the logistic regression model inside a pipeline with scaling and increased iterations
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
model.fit(X, y)

# Save the model to a file using joblib
project_root = get_project_root()
model_dir = os.path.join(project_root, 'model')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'logistic_regression_model.pkl')
joblib.dump(model, model_path)

print("Model has been trained and saved!")