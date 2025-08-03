import transformers
import random
import numpy as np
import os
import mlflow


def set_seed(seed: int):
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def mlflow_set_experiment(experiment_name: str):
    """Set the MLflow experiment name."""
    mlflow.set_experiment(experiment_name)


def mlflow_start_run(run_name: str = None):
    """Start an MLflow run (optionally with a run name)."""
    mlflow.start_run(run_name=run_name)
