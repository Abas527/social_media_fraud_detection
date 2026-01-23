import mlflow
import mlflow.sklearn
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_mlflow(experiment_name="social_media_fraud_detection"):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow tracking URI set to: ./mlruns")
    logger.info(f"MLflow experiment set to: {experiment_name}")

def log_metrics(metrics_dict, step=None):
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(metric_name, metric_value, step=step)
        
def log_params(params_dict):
    for param_name, param_value in params_dict.items():
        mlflow.log_param(param_name, param_value)
