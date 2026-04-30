import keras
import numpy as np
import logging
import pandas as pd

from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from .basedetector_interface import BaseAnomalyDetector
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score,
    roc_auc_score, 
    roc_curve, 
    precision_recall_curve
)


class AutoEncoder(BaseAnomalyDetector):
    
    def __init__(
            self,
            model_core: keras.Model = None,
            threshold: float = None,
            model_name: str="test_model"):
        
        self.model_core = model_core
        self.threshold = threshold

        self.history = None
        self.model_name = model_name

# ======================================================
    def get_model_core(self) -> keras.Model:
        return self.model_core

# ======================================================
    def set_model_core(self, model: keras.Model) -> bool:
        self.model_core = model
        return True

# ======================================================
    def build_model(self, input_dim: int=26)->keras.Model:
        model = keras.Sequential([
            keras.layers.Dense(input_dim, activation='elu', input_shape=(input_dim,)),
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(10, activation='elu'),
            keras.layers.Dense(16, activation='elu'),
            keras.layers.Dense(input_dim, activation='elu')
        ])

        model.compile(
            optimizer="adam", 
            loss="mse", 
            metrics=[MeanAbsoluteError(), RootMeanSquaredError(name="rmse")])
        
        return model

# ======================================================
    def fit(self,
            
            X_train: np.ndarray,
            X_test: np.ndarray,

            X_val: np.ndarray,
            Y_val: np.ndarray,
            
            epochs: int=5,
            batch_size: int=80) -> Dict[keras.Model, Any]:
        
        # Если self.model = None то создаем базовую моель через build_model
        if self.model_core is None:
            self.model_core = self.build_model()
            logging.info(f"An empty model upon initialization. The model was created automatically. \n{self.model_core.summary()}")

        
        # Обучаем корневую модель
        self.history = self.model_core.fit(
            X_train, 
            X_train,
            validation_data=(X_test, X_test),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1)
        logging.info(f"Model training completed")

        # Подбираем порог аномалии
        self.threshold = self.choose_optimal_threshold(
            model=self.model_core,

            X_val=X_val, 
            y_val=Y_val,
        )
        logging.info(f"Threshold selection is complete. \nThreshold is {self.threshold}")
        
        train_result = {
            "threshold": self.threshold,
            "history": self.history.history,
            "epochs": epochs,
            "batch_size": batch_size
        }

        return train_result

# ======================================================
    def predict(
            self, 
            X: np.ndarray, 
            threshold: Optional[float]):

            '''
            Возвращает бинарные предсказания (1 = норма, 0 = аномалия).

            Parameters
            ----------
            X : np.ndarray
                Данные для предсказания формы (n_samples, n_features).
            threshold : float, optional
                Порог классификации. Если None, используется сохранённый `_threshold`.

            Классификация: больше порога = аномалия (0), меньше = норма (1)
            # (предполагаем, что higher score = more anomalous)

            Returns
            -------
            predictions : np.ndarray
                Бинарные предсказания (1 = норма, 0 = аномалия).
            '''

            scores = self.predict_scores(X)
            predictions = (scores < threshold).astype(int)  # 1 = норма, 0 = аномалия
            
            return predictions
    
# ======================================================
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        '''
        Возвращает матрицу рконтрукций
        '''
        X_recon = self.model_core.predict(X, verbose=0)
        logging.info(f"The reconstruction is complete")

        res = np.mean(np.square(X - X_recon), axis=1)
        logging.info(f"MCE calculation completed")

        return res
    

# ======================================================
    def choose_optimal_threshold(
            self,
            model: keras.Model,

            X_val: pd.DataFrame, 
            y_val: pd.Series,

            feature_names: list = None,
            metric: str = 'f1',  # 'f1', 'precision', 'recall', 'balanced'
            target_recall: float = 0.95,  # для стратегии 'recall'
            
            plot: bool = True,
            run_id: str = None) -> dict:
        """
        Подбирает оптимальный порог реконструкционной ошибки (MSE) на валидационной выборке.
        
        Работает с данными из split_data_by_engine: X_val содержит и норму, и аномалию.
        
        Parameters
        ----------
        model : keras.Model
            Обученный автоэнкодер.
        X_val : pd.DataFrame
            Валидационные данные (признаки). Должны быть уже нормализованы.
        y_val : pd.Series
            Метки валидационных данных ('Norm'/'Anom' или 0/1).
        feature_names : list, optional
            Список колонок-сенсоров. Если None, берутся все числовые колонки.
        metric : str
            Стратегия выбора порога:
            - 'f1': максимизировать F1-score
            - 'precision': максимизировать Precision при Recall >= target_recall
            - 'recall': максимизировать Recall при Precision >= 0.5
            - 'balanced': точка, где Precision ≈ Recall
        target_recall : float
            Целевой уровень полноты для стратегий 'precision'/'recall'.
        plot : bool
            Построить график распределения ошибок и ROC-кривую.
        run_id : str, optional
            Идентификатор эксперимента для подписи графиков.
            
        Returns
        -------
        dict : {
            'threshold': float,          # выбранный порог
            'metrics': dict,             # все метрики на валидации
            'results_df': pd.DataFrame,  # детали по каждому образцу
            'plot_path': str or None     # путь к сохраненному графику
        }
        """
        
        logging.info(f"=== START CHOOSE THRESHOLD ===")

        # ======================================================
        # 1. Подготовка данных
        # ======================================================
        # Если передан DataFrame, берем только нужные фичи
        logging.info(f"Deleted non sensors columns...")
        if feature_names is None:
            # Автоматически исключаем не-сенсоры
            exclude = ['unit_number', 'cycle', 'label', 'is_anom', 'RUL']
            feature_names = [c for c in X_val.columns if c not in exclude and np.issubdtype(X_val[c].dtype, np.number)]
            logging.info(f"[Auto] Features selected: {len(feature_names)}")
        
        X_val_features = X_val[feature_names].values

        
        # Нормализация меток: приводим к бинарному виду (1 = норма, 0 = аномалия)
        # Поддерживаем разные форматы: 'Norm'/'Anom', 1/0, True/False
        # if y_val.dtype == object or y_val.dtype == str:
        #     y_val_binary = (y_val == self.split_info.get('normal_label', 'Norm')).astype(int)
        # else:
        #     # Если уже числа, предполагаем 1=норма, 0=аномалия (как в исходном коде)
        y_val_binary = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
        
        logging.info(f"Validation: {len(X_val_features)} samples, Norm: {y_val_binary.sum()}, Anom: {(1-y_val_binary).sum()}")
        logging.info(f"Validation: {len(X_val_features)} samples, Norm: {y_val_binary}, Anom: {(1-y_val_binary)}")
        

        # ======================================================
        # 2. Предсказание и расчет ошибки 
        # ======================================================

        # Получаем матрицу восстановленных значений
        X_val_recon = model.predict(X_val_features, verbose=0)
        logging.info(f"X_val_features:\n{X_val_features}")
        logging.info(f"X_val_recon:\n{X_val_recon}")
        
        # Для алгоритма z1_core этот этап пропускаем
        # if X_val_features.shape == X_val_recon.shape and X_val_recon.shape == 2 and X_val_recon.shape == 2:
        #     mtk_errors = np.mean(np.square(X_val_features - X_val_recon), axis=1)
        # else:
        #     mtk_errors = X_val_recon
        
        # Получаем MSE-ошибки восстановления (вектор)
        mtk_errors = np.nanmax(np.square(X_val_features - X_val_recon), axis=1)
        logging.info(f"Reconstruction mse_errors:\n{mtk_errors}")

        # ======================================================
        # 3. Сбор результатов
        # ======================================================
        results_df = pd.DataFrame({
            'mse': mtk_errors,
            'true_class': y_val_binary,  # 1 = норма, 0 = аномалия
            'true_label': y_val.values if hasattr(y_val, 'values') else y_val  # оригинальная метка для отладки
        })
        
        
        # ======================================================
        # 4. Перебор порогов 
        # ======================================================
        # Оптимизация: берем не все уникальные MSE, а перцентили для скорости
        candidate_thresholds = np.percentile(mtk_errors, np.linspace(0, 100, 500))
        logging.info(f"Candidate thresholds:\n{candidate_thresholds}")

        
        best_threshold = None
        best_score = -1
        all_scores = []  # для графика
        
        for thr in candidate_thresholds:
            # Предсказание: MSE < порог → норма (1), иначе аномалия (0)
            pred_class = (mtk_errors < thr).astype(int)
            
            # Защита от деления на ноль и пустых предсказаний
            if pred_class.sum() == 0 or pred_class.sum() == len(pred_class):
                continue
                
            precision = precision_score(y_val_binary, pred_class, zero_division=0)
            recall = recall_score(y_val_binary, pred_class, zero_division=0)
            f1 = f1_score(y_val_binary, pred_class, zero_division=0)
            
            # Выбор стратегии
            if metric == 'f1':
                score = f1
            elif metric == 'precision':
                score = precision if recall >= target_recall else -1
            elif metric == 'recall':
                score = recall if precision >= 0.5 else -1
            elif metric == 'balanced':
                # Минимизируем разницу между Precision и Recall
                score = 1 - abs(precision - recall) if min(precision, recall) > 0 else -1
            else:
                score = f1  # fallback
            
            all_scores.append({'threshold': thr, 'precision': precision, 'recall': recall, 'f1': f1, 'score': score})
            
            if score > best_score:
                best_score = score
                best_threshold = thr
        
        if best_threshold is None:
            raise ValueError("Unable to find threshold. Please check your data and metrics..")
        

        return best_threshold
    
        # # ======================================================
        # # 5. Финальные метрики
        # # ======================================================
        # final_pred = (mtk_errors < best_threshold).astype(int)
        # final_metrics = {
        #     'precision': precision_score(y_val_binary, final_pred, zero_division=0),
        #     'recall': recall_score(y_val_binary, final_pred, zero_division=0),
        #     'f1': f1_score(y_val_binary, final_pred, zero_division=0),
        #     'accuracy': (final_pred == y_val_binary).mean(),
        #     'roc_auc': roc_auc_score(y_val_binary, -mtk_errors),  # инвертируем, т.к. меньше MSE = лучше
        #     'threshold': best_threshold,
        #     'n_predictions': {
        #         'predicted_normal': int((final_pred == 1).sum()),
        #         'predicted_anomaly': int((final_pred == 0).sum()),
        #         'true_normal': int(y_val_binary.sum()),
        #         'true_anomaly': int((1 - y_val_binary).sum())
        #     }
        # }
        
        # results_df['pred_class'] = final_pred
        # results_df['is_correct'] = (final_pred == y_val_binary).astype(int)
        
        # logging.info(f"Threshold: {best_threshold:.6f}")
        # logging.info(f"Metrics: F1={final_metrics['f1']:.4f}, Prec={final_metrics['precision']:.4f}, Rec={final_metrics['recall']:.4f}")
        
        # # --- 6. Визуализация (опционально, для статьи) ---
        # # plot_path = None
        # # if plot:
        # #     plot_path = self._plot_threshold_analysis(
        # #         results_df, final_metrics, best_threshold,
        # #         metric, run_id or 'threshold_analysis'
        # #     )
        
        
        
        
        # return {
        #     'threshold': float(best_threshold),
        #     'metrics': final_metrics,
        #     'results_df': results_df,
        #     'score_history': pd.DataFrame(all_scores)
        #     # 'plot_path': plot_path
        # }
