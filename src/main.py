# ======================================================
# Главный модуль
# ======================================================

from .training.trainer import train_model, compare_weights
from .config import (
    setup_mlflow, 
    PATH_TRAIN_RAW,
    PATH_TRAIN_FINAL,
    PATH_TRAIN_PROCESSED,
    PATH_TRAIN_FINAL,
    PATH_TRAIN_ADD_RAW,
    PATH_TRAIN_ADD_FINAL,
    
    PATH_LOG,
    PATH_SKALERS,
    
    MLFLOW_TRACKING_URI,
    MLFLOW_REPO_OWNER,
    MLFLOW_REPO_NAME,
    MLFLOW_USERNAME,
    )



if __name__ == "__main__":
    # Настройка MLflow
    setup_mlflow(
        repo_owner = MLFLOW_REPO_OWNER,
        repo_name = MLFLOW_REPO_NAME,
        tracking_uri = MLFLOW_TRACKING_URI,
        username = MLFLOW_USERNAME
    )


