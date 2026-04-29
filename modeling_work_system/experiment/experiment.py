import mlflow
import mlflow.keras
import dagshub
import numpy as np
import keras
import logging
import pandas as pd
import tempfile
import os
import json

# from numpy import load_csv_to_numpy
from pathlib import Path
from mlflow.models import infer_signature
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    accuracy_score, 
    confusion_matrix
)


class Experiment:

    """
    Класс для произведения экспериментов
    Поддерживает методы для проведения эксперимента, что включает в себя:
     1. Обучение модели
     2. Сохранение логов в mlflow
     3. Выгрузку модели из mlflow
    """

    def __init__(
            self,
            # mlflow_tracking_uri: str, 
            # mlflow_repo_owner: str, 
            # mlflow_repo_name: str, 
            # mlflow_username: str,
            # mlflow_pass: str,
            # mlflow_token: str,

            # train_data: dict[str, np.ndarray],

            # # epochs: int = 3,
            # # batch_size: int = 80,

            # model_type="Autoencoder",
            # # model_name: str = "test_model",
            experiment_name: str = "Autoencoder_Anomaly_v2"
            ):
    
        # Иницианилзируем данные для подключения к dugshub
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = mlflow_pass
        os.environ['MLFLOW_TRACKING_TOKEN'] = mlflow_token

        dagshub.auth.add_app_token(token = mlflow_token)
        dagshub.init(
            repo_owner = mlflow_repo_owner, 
            repo_name = mlflow_repo_name, 
            mlflow = True
            )
        
        # dagshub.mlflow.set_tracking_uri(mlflow_tracking_uri)
        logging.info("--- DUGSHUB-MLFLOW CONFIGURATION COMPLETE ---")

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_repo_owner = mlflow_repo_owner
        self.mlflow_repo_name = mlflow_repo_name
        self.mlflow_username = mlflow_username
        self.mlflow_pass = mlflow_pass
        self.mlflow_token = mlflow_token

        self.train_data = train_data
        
        self.epochs = epochs
        self.batch_size = batch_size

        self.model_type = model_type
        self.model_name = model_name
        self.experiment_name = experiment_name
        logging.info("--- EXPERIMENT CONFIGURATION COMPLETE ---")

        return None

# ======================================================
    def send_experiment_to_mlflow_new(
        self,
        model,
        training_history: dict,

        split_data: dict,
        threshold_result: dict,
        
        feature_names: list = None,

        additional_params: dict = None,
        log_predictions: bool = False,
        max_samples_log: int = 100
    ):
        """
        Логирует обученную модель, метрики и артефакты в MLflow.
        
        Работает с новым форматом данных: split_data + threshold_result.
        
        Parameters
        ----------
        model : keras.Model
            Обученная модель автоэнкодера.
        training_history : dict
            История обучения с ключами 'loss', 'val_loss'.
        split_data : dict
            Результат split_data_by_engine с ключами:
            'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test', 'info'.
        threshold_result : dict
            Результат choose_optimal_threshold с ключами:
            'threshold', 'metrics', 'results_df', 'plot_path'.
        experiment_name : str
            Имя эксперимента в MLflow.
        registered_model_name : str
            Имя для регистрации модели в Model Registry.
        epochs : int
            Количество эпох обучения.
        batch_size : int
            Размер батча.
        feature_names : list, optional
            Список имен признаков (сенсоров).
        additional_params : dict, optional
            Дополнительные гиперпараметры для логирования.
        log_predictions : bool
            Логировать предсказания для тестовых образцов (осторожно: большой объем).
        max_samples_log : int
            Максимальное число сэмплов для логирования предсказаний.
            
        Returns
        -------
        str
            ID запуска (run_id) в MLflow.
        """
       
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(self.experiment_name)
        
        with mlflow.start_run(run_name=f"{self.model_name}_run") as run:
            
            # ==================== ПАРАМЕТРЫ ЭКСПЕРИМЕНТА ====================
            mlflow.log_param("model_type", "Autoencoder")
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("input_dim", split_data['X_train'].shape[1])
            mlflow.log_param("anomaly_label", split_data['info'].get('anomaly_label', 'Anom'))
            mlflow.log_param("normal_label", split_data['info'].get('normal_label', 'Norm'))
            
            # Статистика сплита
            mlflow.log_param("n_train_engines", split_data['info']['n_train_units'])
            mlflow.log_param("n_val_engines", split_data['info']['n_val_units'])
            mlflow.log_param("n_test_engines", split_data['info']['n_test_units'])
            mlflow.log_param("n_train_samples", split_data['info']['n_train_samples'])
            mlflow.log_param("n_val_samples", split_data['info']['n_val_samples'])
            mlflow.log_param("n_test_samples", split_data['info']['n_test_samples'])
            
            # Признаки
            if feature_names:
                mlflow.log_param("n_features", len(feature_names))
                # Логируем список фич как артефакт (JSON)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(feature_names, f)
                    mlflow.log_artifact(f.name, artifact_path="metadata")
                    os.unlink(f.name)
            
            # Дополнительные параметры
            if additional_params:
                for key, value in additional_params.items():
                    mlflow.log_param(key, value)
            
            # ======================================================
            # ================== МЕТРИКИ ОБУЧЕНИЯ ==================
            # ======================================================
            # История по эпохам
            if training_history is not None:
                for epoch, (loss, val_loss) in enumerate(
                    zip(training_history.get("loss", []), training_history.get("val_loss", []))
                ):
                    mlflow.log_metric("train_loss", float(loss), step=epoch)
                    mlflow.log_metric("val_loss", float(val_loss), step=epoch)
                
                # Финальные потери
                if training_history.get("loss"):
                    mlflow.log_metric("final_train_loss", float(training_history["loss"][-1]))
                    mlflow.log_metric("final_val_loss", float(training_history.get("val_loss", [-1])[-1]))
                
            # ======================================================
            # ============= МЕТРИКИ ПОРОГА (VALIDATION) ============
            # ======================================================
            
            if threshold_result is not None:
                threshold_metrics = threshold_result.get('metrics', {})
                
                mlflow.log_metric("optimal_threshold", threshold_result['threshold'])
                mlflow.log_metric("val_f1_score", threshold_metrics.get('f1', 0.0))
                mlflow.log_metric("val_precision", threshold_metrics.get('precision', 0.0))
                mlflow.log_metric("val_recall", threshold_metrics.get('recall', 0.0))
                mlflow.log_metric("val_accuracy", threshold_metrics.get('accuracy', 0.0))
                mlflow.log_metric("val_roc_auc", threshold_metrics.get('roc_auc', 0.0))
            
                # Статистика предсказаний на валидации
                n_preds = threshold_metrics.get('n_predictions', {})
                mlflow.log_metric("val_pred_normal", n_preds.get('predicted_normal', 0))
                mlflow.log_metric("val_pred_anomaly", n_preds.get('predicted_anomaly', 0))
                mlflow.log_metric("val_true_normal", n_preds.get('true_normal', 0))
                mlflow.log_metric("val_true_anomaly", n_preds.get('true_anomaly', 0))

            # ======================================================
            # ================== МЕТРИКИ НА ТЕСТЕ ==================
            # ======================================================
            # Пересчитываем метрики на тесте с использованием подобранного порога
            X_test_features = split_data['X_test'][feature_names].values if feature_names else split_data['X_test'].values
            X_test_recon = model.predict(
                X_test_features 
                # verbose=0
                )

            if X_test_features.shape == X_test_recon.shape and X_test_features.shape == 2 and X_test_recon.shape == 2:
                test_mse = np.max(np.square(X_test_features - X_test_recon), axis=1)
            else:
                test_mse = X_test_recon
            # test_mse = np.nanmax(np.square(X_test_features - X_test_recon), axis=1)

            # Бинаризация меток для теста
            y_test_true = (split_data['y_test'] == split_data['info']['normal_label']).astype(int)

            if threshold_result is not None:
                y_test_pred = (test_mse < threshold_result['threshold']).astype(int)
            else:
                y_test_pred = (test_mse < np.percentile(test_mse, 95).astype(int))
            
 
            test_metrics = {
                'test_f1': f1_score(y_test_true, y_test_pred, zero_division=0),
                'test_precision': precision_score(y_test_true, y_test_pred, zero_division=0),
                'test_recall': recall_score(y_test_true, y_test_pred, zero_division=0),
                'test_accuracy': accuracy_score(y_test_true, y_test_pred),
                'test_roc_auc': roc_auc_score(y_test_true, -test_mse),
                'test_rmse': float(keras.metrics.RootMeanSquaredError()(
                    X_test_features, X_test_recon).numpy())
            }
            
            # Логируем тестовые метрики
            for name, value in test_metrics.items():
                mlflow.log_metric(name, float(value))
            
            # # Confusion matrix как артефакт
            # cm = confusion_matrix(y_test_true, y_test_pred)
            # cm_df = pd.DataFrame(cm, columns=['Pred_Anom', 'Pred_Norm'], 
            #                     index=['True_Anom', 'True_Norm'])
            
            # with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            #     cm_df.to_csv(f.name)
            #     mlflow.log_artifact(f.name, artifact_path="test_results")
            #     os.unlink(f.name)

            # ======================================================
            # ======================= МОДЕЛЬ =======================
            # ======================================================
            # Сигнатура модели (на небольшом подмножестве для скорости)
            sample_size = min(10, len(split_data['X_train']))
            X_sample = split_data['X_train'][feature_names].values[:sample_size] if feature_names else split_data['X_train'].values[:sample_size]
            signature = infer_signature(X_sample, model.predict(X_sample, verbose=0))
            
            # Работет тоько для моделей  [keras|sclearn]
            mlflow.keras.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.model_name,
                signature=signature
                # input_example=X_sample[:1]  # Пример входа для Model Registry
            )
            
            # ======================================================
            # ====================== АРТЕФАКТЫ =====================
            # ======================================================
            with tempfile.TemporaryDirectory() as tmp_dir:
                
                # 1. Результаты подбора порога (детали по каждому образцу)
                results_path = os.path.join(tmp_dir, "threshold_analysis.csv")
                threshold_result['results_df'].to_csv(results_path, index=False)
                mlflow.log_artifact(results_path, artifact_path="threshold_analysis")
                
                # 2. График анализа порога (если был сгенерирован)
                if threshold_result.get('plot_path') and os.path.exists(threshold_result['plot_path']):
                    mlflow.log_artifact(threshold_result['plot_path'], artifact_path="plots")
                
                # 3. История скоринга порогов (для графика "метрики vs порог")
                if 'score_history' in threshold_result:
                    history_path = os.path.join(tmp_dir, "threshold_search_history.csv")
                    threshold_result['score_history'].to_csv(history_path, index=False)
                    mlflow.log_artifact(history_path, artifact_path="threshold_analysis")
                
                # 4. Конфиг эксперимента (воспроизводимость)
                config = {
                    "experiment_name": self.experiment_name,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "split_info": {k: v for k, v in split_data['info'].items() 
                                  if not isinstance(v, (np.ndarray, list))},
                    "threshold_strategy": threshold_result.get('strategy', 'f1'),
                    "feature_names": feature_names[:10] + ['...'] if feature_names and len(feature_names) > 10 else feature_names
                }
                config_path = os.path.join(tmp_dir, "experiment_config.json")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
                mlflow.log_artifact(config_path, artifact_path="metadata")
                
                # 5. (Опционально) Предсказания для тестовых сэмплов
                if log_predictions:
                    n_log = min(max_samples_log, len(test_mse))
                    pred_df = pd.DataFrame({
                        'mse': test_mse[:n_log],
                        'true_label': split_data['y_test'].values[:n_log],
                        'true_binary': y_test_true[:n_log],
                        'pred_binary': y_test_pred[:n_log],
                        'is_correct': (y_test_true[:n_log] == y_test_pred[:n_log]).astype(int)
                    })
                    if hasattr(split_data['X_test'], 'iloc'):
                        # Добавляем несколько сенсоров для отладки
                        for feat in feature_names[:3] if feature_names else split_data['X_test'].columns[:3]:
                            pred_df[f'sensor_{feat}'] = split_data['X_test'][feat].values[:n_log]
                    
                    pred_path = os.path.join(tmp_dir, "test_predictions_sample.csv")
                    pred_df.to_csv(pred_path, index=False)
                    mlflow.log_artifact(pred_path, artifact_path="test_results")
            
            # ======================================================
            # =================== ТЕГИ ДЛЯ ПОИСКА ==================
            # ======================================================

            mlflow.set_tag("mlflow.runName", f"{self.model_name}_run")
            mlflow.set_tag("task", "anomaly_detection")
            mlflow.set_tag("dataset", "NASA_CMAPSS")
            if threshold_metrics.get('f1', 0) > 0.8:
                mlflow.set_tag("quality", "high")
            
            logging.info(f"✓ The experiment is logged in MLflow. Run ID: {run.info.run_id}")
            logging.info(f"✓ Test F1: {test_metrics['test_f1']:.4f}, Recall: {test_metrics['test_recall']:.4f}")
            
            return run.info.run_id

# ======================================================
    def send_experiment_to_mlflow_mini(
        self,
        model,
        training_history: dict,
        threshold_result: dict = None,

        X_train: np.ndarray = None,
        Y_train: np.ndarray = None,

        X_test: np.ndarray = None,
        Y_test: np.ndarray = None,

        X_val: np.ndarray = None,
        Y_val: np.ndarray = None,

        f1_score: float | np.ndarray = None,
        precision_score = None,
        recall_score = None,
        accuracy_score = None,
        roc_auc_score = None,
        rmse_score = None
    ):
        
        """
        Простой метод логирования эксперимента в MLflow.
        Совместим с Python 3.10+ и MLflow >= 2.10
        """
        
        # Устанавливаем эксперимент
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=f"{self.model_name}_run") as run:
            
            mlflow.keras.log_model(model=model,  artifact_path="model")
            try:
                print(f"🔄 Логирование модели: {type(model)}")
                mlflow.keras.log_model(
                    model=model,
                    artifact_path="model",  # ← это обязательно!
                    registered_model_name=self.model_name
                    # signature=mlflow.models.infer_signature(X_train, model.predict(X_train[:100])) if X_train is not None else None,
                    # input_example=X_train[:1] if X_train is not None else None,
                )
                print("✅ Модель успешно залогирована в папку 'model'")
            except Exception as e:
                print(f"⚠️ Ошибка при логировании модели: {e}")
                #Фолбэк: сохраняем модель вручную в файл и логируем как артефакт
                fallback_path = Path("artifacts") / "model_fallback.keras"
                fallback_path.parent.mkdir(exist_ok=True)
                model.save(str(fallback_path))  # нативный save Keras
                mlflow.log_artifact(str(fallback_path), artifact_path="artifacts")
                print(f"💾 Модель сохранена как артефакт: {fallback_path}")


            # mlflow.log_metric("train_loss", training_history)
            for epoch, (loss, val_loss) in enumerate(
                    zip(training_history.get("loss", []), training_history.get("val_loss", []))
                ):
                    mlflow.log_metric("train_loss", float(loss), step=epoch)
                    mlflow.log_metric("val_loss", float(val_loss), step=epoch)

            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("batch_size", self.batch_size)
            # mlflow.log_param("threshold_result", threshold_result)

        
            return run.info.run_id


# ======================================================
    def load_model_from_mlflow(
        self,
        stage: str = "None"  # или "Staging", "None", либо конкретная версия как строка "1"
        ) -> keras.Model:
        
        """
        Загружает модель из MLflow Model Registry.
        
        Parameters
        ----------
        registered_model_name : str
            Имя модели в MLflow Registry (например, "autoencoder_turbo").
        stage : str, optional
            Стадия модели: "Production", "Staging", "None" (последняя версия),
            или номер версии в виде строки, например "3".
        tracking_uri : str, optional
            URI для подключения к MLflow (например, "file:///path/to/mlruns" или "http://localhost:5000").
            Если не указан — используется текущий активный URI.
        
        Returns
        -------
        mlflow.pyfunc.PyFuncModel
            Загруженная модель, готовая к вызову через `.predict(X)`.
        """

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Формируем URI модели в формате MLflow
        model_uri = f"models:/{self.model_name}/latest"

        try:
            model = mlflow.keras.load_model(
                model_uri 
                # compile=False
                )
            logging.info(f"The model is loaded from mlflow: {model_uri}")
            return model
        
        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow by URI '{model_uri}': {e}")

# # ======================================================
#     def train_model(
#         self,
#         model: keras.Model,
#         train_df: np.ndarray,
#         test_df: np.ndarray
#     ) -> keras.Model:
        
#         """Обучает модель автокодировщика на нормальных данных."""
#         history = model.fit(
#             train_df, 
#             train_df,
#             validation_data = (test_df, test_df),
#             epochs = self.epochs,
#             batch_size = self.batch_size,
#             shuffle = True,
#             verbose = 1)

#         return model, history.history

# ======================================================
    # def compare_weights(
    #         self, 
    #         model1: keras.Model, 
    #         model2: keras.Model,
    #         tolerance=1e-5):
    #     weights1 = model1.get_weights()
    #     weights2 = model2.get_weights()
    #     """
    #     Сравнивает веса двух моделей
    #     """
        
    #     if len(weights1) != len(weights2):
    #         print("Модели имеют разное количество слоев с весами")
    #         return False
        
    #     for i, (w1, w2) in enumerate(zip(weights1, weights2)):
    #         if not np.allclose(w1, w2, rtol=tolerance, atol=tolerance):
    #             print(f"Различие в весах на слое {i}")
    #             return False
            
    #     return True