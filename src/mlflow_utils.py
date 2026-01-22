import mlflow
import os

def setup_experiment(experiment_name="social_media_fraud_detection"):
    """Setup MLflow experiment with local tracking URI"""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)