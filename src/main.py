# main.py
from config.mlflow_config import setup_mlflow
from training.trainer import train_and_log_to_mlflow


if __name__ == "__main__":
    # Настройка MLflow
    setup_mlflow(
        repo_owner="Dimitriy200",
        repo_name="diplom_autoencoder",
        tracking_uri="https://dagshub.com/Dimitriy200/diplom_autoencoder.mlflow",
        username="your_username"  # лучше из .env
    )

    # Запуск обучения
    model = train_and_log_to_mlflow(
        train_path="data/train.csv",
        valid_path="data/valid.csv",
        predict_path="data/predict.csv",
        experiment_name="Autoencoder_Experiment_v1",
        registered_model_name="autoencoder_3",
        epochs=10,
        batch_size=80
    )