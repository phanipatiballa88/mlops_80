# FILE: train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load the training data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Train a RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train.values.ravel())

# Create the model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model
joblib.dump(model, 'model.pkl')
