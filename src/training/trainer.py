# training/trainer.py
import mlflow
import numpy as np

from numpy import load_csv_to_numpy
from mlflow.models import infer_signature
from models.autoencoder import create_contractive_autoencoder
from evaluation.metrics import compute_rmse


def train_and_log_to_mlflow(
    train_path: str,
    valid_path: str,
    predict_path: str,
    experiment_name: str,
    registered_model_name: str,
    epochs: int = 5,
    batch_size: int = 80,
    model_type: str = "contractive"
):
    # Загрузка данных
    X_train = load_csv_to_numpy(train_path)
    X_valid = load_csv_to_numpy(valid_path)
    X_pred = load_csv_to_numpy(predict_path)

    # Выбор модели
    if model_type == "contractive":
        model = create_contractive_autoencoder(input_dim=X_train.shape[1])
    else:
        raise ValueError("Unsupported model type")

    # Настройка MLflow
    mlflow.set_experiment(experiment_name)
    mlflow.keras.autolog()

    with mlflow.start_run():
        # Обучение
        history = model.fit(
            X_train, X_train,
            validation_data=(X_valid, X_valid),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True
        )

        # Предсказание
        X_recon = model.predict(X_pred)

        # Метрики
        rmse = compute_rmse(X_pred, X_recon)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("epochs", epochs)

        # Логирование модели
        signature = infer_signature(X_train, model.predict(X_train[:10]))
        mlflow.keras.log_model(
            model,
            artifact_path="autoencoder",
            registered_model_name=registered_model_name,
            signature=signature
        )

    return model