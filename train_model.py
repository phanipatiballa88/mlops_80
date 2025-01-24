# FILE: train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Set the MLflow tracking URI (optional, defaults to ./mlruns)
mlflow.set_tracking_uri("mlruns")

# Set the experiment name
mlflow.set_experiment("mlops")

# Load the training data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')

# Define different sets of parameters for experimentation
params_list = [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 15},
]

for params in params_list:
    with mlflow.start_run():
        # Train a RandomForest model with different parameters
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train.values.ravel())

        # Log model parameters
        mlflow.log_params(params)

        # Log model metrics
        accuracy = model.score(X_train, y_train)
        mlflow.log_metric("accuracy", accuracy)

        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model with signature and input example
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train.head())

        # Create the model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)

        # Save the model
        joblib.dump(model, 'model.pkl')