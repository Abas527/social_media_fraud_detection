import mlflow_utils
import os
import mlflow.sklearn

def setup_experiment(experiment_name="social_media_fraud_detection"):

    mlflow_utils.set_tracking_uri("file:./mlruns")
    mlflow_utils.set_experiment(experiment_name)