import pandas as pd

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict


@dataclass
class AEMetricResult:
    """Контейнер для результатов метрик (удобно для экспорта в MLflow/CSV)."""
    
    # Основные метрики
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0
    
    # Ранжирующие метрики
    roc_auc: float = 0.0
    pr_auc: float = 0.0  # Average Precision
    
    # Статистика предсказаний
    n_samples: int = 0
    n_true_normal: int = 0
    n_true_anomaly: int = 0
    n_pred_normal: int = 0
    n_pred_anomaly: int = 0
    n_true_positive: int = 0  # Правильно предсказанная норма
    n_true_negative: int = 0  # Правильно предсказанная аномалия
    n_false_positive: int = 0  # Ложная тревога
    n_false_negative: int = 0  # Пропущенная аномалия
    
    # Порог
    threshold: float = 0.0
    
    # Дополнительные метрики (для расширения)
    extra_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        """Плоский словарь для логирования в MLflow."""
        base = asdict(self)
        extra = base.pop('extra_metrics')
        return {**base, **extra}
    
    def to_series(self) -> pd.Series:
        """Для экспорта в DataFrame."""
        return pd.Series(self.to_dict())