import numpy as np
import pandas as pd
import logging

from sklearn.metrics import (roc_auc_score, mean_squared_error, f1_score, accuracy_score, 
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix)
from typing import Dict, List, Optional, Any


def run_reconstruction_comparison_table(
    models: Dict[str, Any],
    norm_engines: pd.DataFrame,
    anom_engines: pd.DataFrame,
    engine_col: str = 'unit number',
    features: Optional[List[str]] = None,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42,
    return_raw: bool = False
) -> pd.DataFrame:
    """
    Сравнивает модели через кластерный бутстрап по двигателям.
    Принимает pd.DataFrame, автоматически группирует по engine_col.
    Метрики: ROC-AUC и RMSE.
    """
    rng = np.random.default_rng(seed)
    
    # 1️⃣ Определяем признаки
    if features is None:
        features = [c for c in norm_engines.columns]
        anom_features = [c for c in anom_engines.columns]
        if set(features) != set(anom_features):
            raise ValueError("Несовпадение признаков в norm_engines и anom_engines")
    
    # 2️⃣ Группируем по двигателям ОДИН раз
    norm_groups = {name: group[features].values for name, group in norm_engines.groupby(engine_col)}
    anom_groups = {name: group[features].values for name, group in anom_engines.groupby(engine_col)}
    
    norm_ids = list(norm_groups.keys())
    anom_ids = list(anom_groups.keys())
    
    if not norm_ids or not anom_ids:
        raise ValueError("Не найдены двигатели в одном из датафреймов")
    
    n_eng_norm = len(norm_ids)
    n_eng_anom = len(anom_ids)
    
    logging.info(f"Cluster bootstrap: {n_eng_norm} normal engines, {n_eng_anom} anomal")
    
    # Хранилище сырых результатов
    raw_records = []
    
    for b in range(n_bootstrap):
        # 1️⃣ Кластерный бутстрап: выбираем ID двигателей с возвращением
        sampled_norm_ids = rng.choice(norm_ids, size=n_eng_norm, replace=True)
        sampled_anom_ids = rng.choice(anom_ids, size=n_eng_anom, replace=True)
        
        # 2️⃣ Склеиваем целые двигатели в единые массивы
        X_norm_bs = np.concatenate([norm_groups[eid] for eid in sampled_norm_ids])
        X_anom_bs = np.concatenate([anom_groups[eid] for eid in sampled_anom_ids])
        
        # 3️⃣ Предсказания и метрики для каждой модели
        for name, model in models.items():
            # Предполагаем, что reconstruct() возвращает реконструкцию
            X_rec_norm = model.reconstruct(X_norm_bs)
            X_rec_anom = model.reconstruct(X_anom_bs)
            
            # Ошибки реконструкции по объектам
            err_norm = np.mean((X_norm_bs - X_rec_norm) ** 2, axis=1)
            err_anom = np.mean((X_anom_bs - X_rec_anom) ** 2, axis=1)

            rmse_norm = np.sqrt(np.mean(err_norm))
            rmse_anom = np.sqrt(np.mean(err_anom))

            rmse_gap = rmse_anom - rmse_norm
            rmse_ratio = rmse_anom / rmse_norm if rmse_norm > 1e-8 else np.nan
            
            # y_true = np.concatenate([np.zeros(len(err_norm)), np.ones(len(err_anom))])
            # y_scores = np.concatenate([err_norm, err_anom])
            y_true = np.concatenate([np.zeros(len(err_norm)), np.ones(len(err_anom))])
            y_scores = np.concatenate([err_norm, err_anom])
            roc_auc = roc_auc_score(y_true, y_scores)
            
            rec = {
                'model': name,
                'iteration': b,
                # Раздельные метрики
                'rmse_norm': rmse_norm,
                'rmse_anom': rmse_anom,
                'rmse_gap': rmse_gap,
                'rmse_ratio': rmse_ratio,
                # Общая метрика качества разделения
                'roc_auc': roc_auc,
                # Дополнительно: средние ошибки (не квадратичные)
                'mean_err_norm': np.mean(err_norm),
                'mean_err_anom': np.mean(err_anom),
            }
                
            raw_records.append(rec)
            
    raw_df = pd.DataFrame(raw_records)
    
    # 📊 Агрегация статистик
    metrics_cols = [
        'rmse_norm', 'rmse_anom', 'rmse_gap', 'rmse_ratio',
        'roc_auc', 'mean_err_norm', 'mean_err_anom'
        ]
    alpha = 1 - confidence_level
    ci_low_pct = (alpha / 2) * 100
    ci_high_pct = (1 - alpha / 2) * 100
    
    summary_data = {}
    for name in models.keys():
        model_vals = raw_df[raw_df['model'] == name][metrics_cols]
        model_stats = {}
        
        for col in metrics_cols:
            vals = model_vals[col].dropna()
            if len(vals) == 0:
                model_stats[(col, 'mean')] = np.nan
                model_stats[(col, 'std')] = np.nan
                model_stats[(col, 'ci_low')] = np.nan
                model_stats[(col, 'ci_high')] = np.nan
                model_stats[(col, 'se')] = np.nan
            else:
                model_stats[(col, 'mean')] = vals.mean()
                model_stats[(col, 'std')] = vals.std(ddof=1)
                model_stats[(col, 'ci_low')] = np.percentile(vals, ci_low_pct)
                model_stats[(col, 'ci_high')] = np.percentile(vals, ci_high_pct)
                model_stats[(col, 'se')] = vals.std(ddof=1) / np.sqrt(len(vals))
                
        summary_data[name] = model_stats
        
    # Формируем DataFrame с мультииндексом
    cols = pd.MultiIndex.from_product([metrics_cols, ['mean', 'std', 'ci_low', 'ci_high', 'se']])
    summary_df = pd.DataFrame(summary_data).T.reindex(columns=cols)
    summary_df.index.name = 'model'

    logging.info(f"raw_df: \n{raw_df}")
    logging.info(f"summary_df: \n{summary_df}")
    
    return (summary_df, raw_df) if return_raw else summary_df





def run_classification_comparison_table(
    models: Dict[str, Any],
    norm_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    engine_col: str,
    features: Optional[List[str]] = None,
    pos_label: int = 1,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    seed: int = 42,
    return_raw: bool = False
) -> pd.DataFrame:
    """
    Сравнивает модели на этапе классификации через кластерный бутстрап.
    Модели должны иметь метод .predict(X), возвращающий бинарные классы (0/1).
    
    Parameters
    ----------
    models : dict
        {имя_модели: объект_модели}
    norm_df, anom_df : pd.DataFrame
        Датафреймы с данными. Должны содержать engine_col и признаки.
    engine_col : str
        Имя столбца с идентификатором двигателя.
    features : list, optional
        Список столбцов-признаков. Если None, берутся все кроме engine_col.
    pos_label : int
        Какой класс считать "позитивным" (по умолчанию 1 = аномалия).
        Precision/Recall/F1 считаются для этого класса.
    n_bootstrap : int
        Количество бутстрап-итераций.
    confidence_level : float
        Уровень доверительного интервала.
    seed : int
        Сид для воспроизводимости.
    return_raw : bool
        Если True, возвращает (summary_df, raw_df).
    """
    rng = np.random.default_rng(seed)
    
    # 1 Определяем признаки
    if features is None:
        features = [c for c in norm_df.columns if c != engine_col]
        anom_features = [c for c in anom_df.columns if c != engine_col]
        if set(features) != set(anom_features):
            raise ValueError("Несовпадение признаков в norm_df и anom_df")
            
    # 2 Группируем по двигателям ОДИН раз
    norm_groups = {name: group[features].values for name, group in norm_df.groupby(engine_col)}
    anom_groups = {name: group[features].values for name, group in anom_df.groupby(engine_col)}
    
    norm_ids = list(norm_groups.keys())
    anom_ids = list(anom_groups.keys())
    
    if not norm_ids or not anom_ids:
        raise ValueError("Не найдены двигатели в одном из датафреймов")
        
    n_eng_norm = len(norm_ids)
    n_eng_anom = len(anom_ids)
    
    raw_records = []
    
    # 3 Бутстрап-цикл
    for b in range(n_bootstrap):
        # Выбираем ID двигателей с возвращением
        sampled_norm_ids = rng.choice(norm_ids, size=n_eng_norm, replace=True)
        sampled_anom_ids = rng.choice(anom_ids, size=n_eng_anom, replace=True)
        
        X_norm_bs = np.concatenate([norm_groups[eid] for eid in sampled_norm_ids])
        X_anom_bs = np.concatenate([anom_groups[eid] for eid in sampled_anom_ids])
        
        # 4 Предсказания и метрики для каждой модели
        for name, model in models.items():
            #  Модель возвращает бинарные классы (0/1)
            y_pred_norm = model.predict(X_norm_bs)
            y_pred_anom = model.predict(X_anom_bs)
            
            y_true = np.concatenate([np.zeros(len(y_pred_norm)), np.ones(len(y_pred_anom))])
            y_pred = np.concatenate([y_pred_norm, y_pred_anom])
            
            # Базовые метрики классификации
            rec = {
                'model': name,
                'iteration': b,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
                'recall': recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
                'f1': f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
                'mcc': matthews_corrcoef(y_true, y_pred),
                'specificity': _specificity(y_true, y_pred, pos_label=pos_label),
            }
            
            # Confusion Matrix (нормализованная по строкам для интерпретируемости)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            total = len(y_true)
            rec['tn_rate'] = tn / total
            rec['fp_rate'] = fp / total
            rec['fn_rate'] = fn / total
            rec['tp_rate'] = tp / total
            
            raw_records.append(rec)
            
    raw_df = pd.DataFrame(raw_records)
    
    # 5 Агрегация статистик
    metrics_cols = [
        'accuracy', 'precision', 'recall', 'f1', 'mcc', 'specificity',
        'tn_rate', 'fp_rate', 'fn_rate', 'tp_rate'
    ]
    alpha = 1 - confidence_level
    ci_low_pct = (alpha / 2) * 100
    ci_high_pct = (1 - alpha / 2) * 100
    
    summary_data = {}
    for name in models.keys():
        model_vals = raw_df[raw_df['model'] == name][metrics_cols]
        model_stats = {}
        
        for col in metrics_cols:
            vals = model_vals[col].dropna()
            if len(vals) == 0:
                model_stats[(col, 'mean')] = np.nan
                model_stats[(col, 'std')] = np.nan
                model_stats[(col, 'ci_low')] = np.nan
                model_stats[(col, 'ci_high')] = np.nan
                model_stats[(col, 'se')] = np.nan
            else:
                model_stats[(col, 'mean')] = vals.mean()
                model_stats[(col, 'std')] = vals.std(ddof=1)
                model_stats[(col, 'ci_low')] = np.percentile(vals, ci_low_pct)
                model_stats[(col, 'ci_high')] = np.percentile(vals, ci_high_pct)
                model_stats[(col, 'se')] = vals.std(ddof=1) / np.sqrt(len(vals))
                
        summary_data[name] = model_stats
        
    cols = pd.MultiIndex.from_product([metrics_cols, ['mean', 'std', 'ci_low', 'ci_high', 'se']])
    summary_df = pd.DataFrame(summary_data).T.reindex(columns=cols)
    summary_df.index.name = 'model'
    
    return (summary_df, raw_df) if return_raw else summary_df


def _specificity(y_true, y_pred, pos_label: int = 1) -> float:
    """
    Specificity = TN / (TN + FP) — доля правильно распознанных негативных случаев (нормы).
    pos_label: какой класс считается "позитивным" (аномалия). Негативный — противоположный.
    """
    neg_label = 1 - pos_label
    tn = np.sum((y_true == neg_label) & (y_pred == neg_label))
    fp = np.sum((y_true == neg_label) & (y_pred == pos_label))
    denom = tn + fp
    return tn / denom if denom > 0 else 0.0





def log_summary_report(summary_df):
    """Красиво выводит таблицу сравнения по каждой метрике."""
    metrics = ['roc_auc', 'rmse_norm', 'rmse_anom', 'rmse_gap', 'rmse_ratio', 
               'mean_err_norm', 'mean_err_anom']
    
    for metric in metrics:
        logging.info(f"\n{'='*60} \nMetric: {metric.upper()} \n{'='*60}")
        
        # Выбираем только колонки этой метрики
        sub_df = summary_df[[c for c in summary_df.columns if c[0] == metric]]
        sub_df.columns = [c[1] for c in sub_df.columns]  # убираем верхний уровень индекса
        
        table_str = sub_df.to_string(float_format=lambda x: f'{x:.4f}')
        for line in table_str.split('\n'):
            logging.info(f"  {line}")
        
        logging.info("")
