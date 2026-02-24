# training/trainer.py
import mlflow
import numpy as np

from numpy import load_csv_to_numpy
from mlflow.models import infer_signature
from models.autoencoder import create_contractive_autoencoder
from evaluation.metrics import compute_rmse


def train_and_log_to_mlflow(
    train_df: np.ndarray,
    test_df: np.ndarray,
    valid_df: np.ndarray,
    experiment_name: str,
    registered_model_name: str,
    epochs: int = 10,
    batch_size: int = 80,
    model_type: str = "contractive"):
    
    # Загрузка данных
    # train_df = load_csv_to_numpy(train_path)
    # test_df = load_csv_to_numpy(valid_path)
    # valid_df = load_csv_to_numpy(predict_path)

    # Выбор модели
    if model_type == "contractive":
        model = create_contractive_autoencoder(input_dim=train_df.shape[1])
    else:
        raise ValueError("Unsupported model type")

    # Настройка MLflow
    mlflow.set_experiment(experiment_name)
    mlflow.keras.autolog()

    with mlflow.start_run():
        # Обучение
        history = model.fit(
            train_df, train_df,
            validation_data=(test_df, test_df),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True
        )

        # Предсказание
        X_recon = model.predict(valid_df)

        # Метрики
        rmse = compute_rmse(valid_df, X_recon)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("epochs", epochs)

        # Логирование модели
        signature = infer_signature(train_df, model.predict(train_df[:10]))
        mlflow.keras.log_model(
            model,
            artifact_path="autoencoder",
            registered_model_name=registered_model_name,
            signature=signature
        )

    return model