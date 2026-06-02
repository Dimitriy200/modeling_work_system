# =====================================
#   СТАТИСТИЧЕССКИЙ АНАЛИЗ МОДЕЛЕЙ
# =====================================
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from scipy import stats
from typing import Dict, List, Any


def paired_t_test(
    df: pd.DataFrame,
    metric: str = "roc_auc",
    model_a: str = "ModelA",
    model_b: str = "ModelB",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Парный t-критерий Стьюдента для сравнения двух моделей по бутстрап-метрикам.
    
    Parameters
    ----------
    df : pd.DataFrame
        Таблица с результатами бутстрапа (столбцы: model, sample_id, <метрики>)
    metric : str
        Название метрики для сравнения (roc_auc, rmse, f1_at_threshold и т.д.)
    model_a, model_b : str
        Имена моделей для попарного сравнения
    alpha : float
        Уровень значимости (по умолчанию 0.05)
        
    Returns
    -------
    dict
        Словарь со статистиками, p-value, размером эффекта и интерпретацией
    """
    # 1 Извлекаем парные массивы
    vals_a = df.loc[df["model"] == model_a, metric].values
    vals_b = df.loc[df["model"] == model_b, metric].values
    
    if len(vals_a) != len(vals_b):
        raise ValueError("Длины выборок не совпадают. Проверьте порядок sample_id.")
    
    # 2 Убираем NaN (сохраняя парность)
    mask = ~np.isnan(vals_a) & ~np.isnan(vals_b)
    vals_a_clean = vals_a[mask]
    vals_b_clean = vals_b[mask]
    
    if len(vals_a_clean) < 2:
        raise ValueError("Недостаточно парных наблюдений для расчёта t-теста.")
        
    # 3 Парный t-тест
    t_stat, p_value = stats.ttest_rel(vals_a_clean, vals_b_clean)
    
    # 4 Размер эффекта (Cohen's d для парных выборок)
    diff = vals_a_clean - vals_b_clean
    std_diff = np.std(diff, ddof=1)
    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0.0
    
    # 5 Интерпретация
    mean_a, mean_b = np.mean(vals_a_clean), np.mean(vals_b_clean)
    winner = model_a if mean_a > mean_b else model_b
    is_significant = p_value < alpha
    
    effect_label = (
        "малый" if abs(cohens_d) < 0.2 else 
        ("средний" if abs(cohens_d) < 0.5 else "большой")
    )
    
    return {
        "metric": metric,
        "model_a": model_a, "mean_a": mean_a, "std_a": np.std(vals_a_clean, ddof=1),
        "model_b": model_b, "mean_b": mean_b, "std_b": np.std(vals_b_clean, ddof=1),
        "winner": winner,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": is_significant,
        "alpha": alpha,
        "cohens_d": float(cohens_d),
        "effect_size": effect_label,
        "n_pairs": len(vals_a_clean)  # ⚠️ это n_bootstrap, а не число двигателей!
    }