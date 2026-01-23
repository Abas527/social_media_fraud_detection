## src/mlflow_utils.py

import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_experiment(experiment_name):
    """
    Setup MLflow experiment with the given name.
    
    Args:
        experiment_name (str): Name of the experiment
    
    Returns:
        str: Experiment ID
    """
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_metrics(metrics_dict):
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
    """
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(metric_name, metric_value)


def log_params(params_dict):
    """
    Log multiple parameters to MLflow.
    
    Args:
        params_dict (dict): Dictionary of parameter names and values
    """
    for param_name, param_value in params_dict.items():
        mlflow.log_param(param_name, param_value)
