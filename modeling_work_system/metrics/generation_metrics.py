import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from sklearn.metrics import pairwise_distances
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import cosine


def calculate_mmd(
    X_real: np.ndarray, 
    X_gen: np.ndarray, 
    kernel: str = 'rbf',
    gamma: Optional[float] = None
) -> float:
    """
    Maximum Mean Discrepancy (MMD) — метрика близости распределений.
    Чем меньше, тем лучше (0 = идентичные распределения).
    
    Args:
        X_real: Реальные данные (N, features) или (N, seq_len, features)
        X_gen: Сгенерированные данные (M, features) или (M, seq_len, features)
        kernel: Тип ядра ('rbf' или 'linear')
        gamma: Параметр для RBF ядра (если None, вычисляется автоматически)
    """
    # Flatten если 3D
    if X_real.ndim == 3:
        X_real = X_real.reshape(X_real.shape[0], -1)
    if X_gen.ndim == 3:
        X_gen = X_gen.reshape(X_gen.shape[0], -1)
    
    n_real, n_gen = X_real.shape[0], X_gen.shape[0]
    
    # Автоматический выбор gamma (median heuristic)
    if gamma is None:
        pairwise_dists = pairwise_distances(X_real, metric='euclidean')
        gamma = 1.0 / (2 * np.median(pairwise_dists) ** 2)
    
    # Вычисление MMD
    if kernel == 'rbf':
        # K(X_real, X_real)
        K_rr = np.exp(-gamma * pairwise_distances(X_real, X_real, metric='sqeuclidean'))
        # K(X_gen, X_gen)
        K_gg = np.exp(-gamma * pairwise_distances(X_gen, X_gen, metric='sqeuclidean'))
        # K(X_real, X_gen)
        K_rg = np.exp(-gamma * pairwise_distances(X_real, X_gen, metric='sqeuclidean'))
        
        mmd = (K_rr.sum() - np.diag(K_rr).sum()) / (n_real * (n_real - 1)) + \
              (K_gg.sum() - np.diag(K_gg).sum()) / (n_gen * (n_gen - 1)) - \
              2 * K_rg.mean()
    else:  # linear
        mmd = np.mean(X_real @ X_real.T) + np.mean(X_gen @ X_gen.T) - \
              2 * np.mean(X_real @ X_gen.T)
    
    return max(0, mmd)  # MMD не должен быть отрицательным


def calculate_diversity(
    X_gen: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Разнообразие сгенерированных данных (среднее попарное расстояние).
    Чем больше, тем лучше (модель не схлопывается в один режим).
    """
    if X_gen.ndim == 3:
        X_gen = X_gen.reshape(X_gen.shape[0], -1)
    
    n_samples = X_gen.shape[0]
    if n_samples < 2:
        return 0.0
    
    # Вычисляем попарные расстояния
    dists = pairwise_distances(X_gen, metric=metric)
    
    # Берем верхний треугольник (без диагонали)
    upper_tri = dists[np.triu_indices(n_samples, k=1)]
    
    return float(upper_tri.mean())


def calculate_coverage(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    k: int = 5
) -> float:
    """
    Coverage — насколько хорошо сгенерированные данные покрывают пространство реальных.
    Доля реальных семплов, у которых хотя бы один из k ближайших соседей в X_gen.
    """
    if X_real.ndim == 3:
        X_real = X_real.reshape(X_real.shape[0], -1)
    if X_gen.ndim == 3:
        X_gen = X_gen.reshape(X_gen.shape[0], -1)
    
    n_real = X_real.shape[0]
    
    # Для каждого реального семпла находим k ближайших в сгенерированных
    dists = pairwise_distances(X_real, X_gen, metric='euclidean')
    
    # Находим минимальное расстояние до X_gen для каждого X_real
    min_dists = dists.min(axis=1)
    
    # Порог: медиана попарных расстояний в X_real
    real_dists = pairwise_distances(X_real, metric='euclidean')
    threshold = np.median(real_dists[np.triu_indices(n_real, k=1)])
    
    # Доля реальных семплов, покрытых X_gen
    coverage = np.mean(min_dists < threshold)
    
    return float(coverage)


def calculate_latent_normality(
    model: Any,
    X_test: np.ndarray,
    device: Any
) -> Dict[str, float]:
    """
    Проверяет, насколько распределение латентных векторов близко к N(0, I).
    Возвращает KS-статистику и p-value для каждого измерения латентного пространства.
    """
    import torch
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        mu, logvar = model.encoder(X_tensor)
        z = model.reparameterize(mu, logvar)
        z_np = z.cpu().numpy()
    
    n_latent = z_np.shape[1]
    ks_stats = []
    p_values = []
    
    # Для каждого измерения латентного пространства
    for i in range(n_latent):
        ks_stat, p_val = ks_2samp(z_np[:, i], np.random.normal(0, 1, size=len(z_np)))
        ks_stats.append(ks_stat)
        p_values.append(p_val)
    
    return {
        'mean_ks_stat': np.mean(ks_stats),
        'mean_p_value': np.mean(p_values),
        'min_p_value': np.min(p_values),
        'n_latent_dims': n_latent
    }


def calculate_reconstruction_quality(
    model: Any,
    X_test: np.ndarray,
    device: Any,
    chunk_size: int = 64
) -> Dict[str, float]:
    """
    Качество реконструкции (MSE, MAE) на тестовых данных.
    """
    import torch
    
    model.eval()
    errors_mse = []
    errors_mae = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), chunk_size):
            batch = torch.tensor(X_test[i:i+chunk_size], dtype=torch.float32).to(device)
            x_recon, mu, logvar = model(batch)
            
            mse = torch.mean((batch - x_recon) ** 2, dim=(1, 2))
            mae = torch.mean(torch.abs(batch - x_recon), dim=(1, 2))
            
            errors_mse.extend(mse.cpu().numpy())
            errors_mae.extend(mae.cpu().numpy())
    
    return {
        'mse_mean': np.mean(errors_mse),
        'mse_std': np.std(errors_mse),
        'mae_mean': np.mean(errors_mae),
        'mae_std': np.std(errors_mae)
    }


def run_generation_comparison_table(
    models: Dict[str, Any],
    X_real_test: np.ndarray,
    device: Any,
    n_generate: int = 100,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    seed: int = 42,
    return_raw: bool = False
) -> pd.DataFrame:
    """
    Сравнивает модели по метрикам качества генерации через бутстрап.
    
    Args:
        models: Словарь {имя_модели: объект_модели}
        X_real_test: Реальные тестовые данные (N, seq_len, features)
        device: torch.device
        n_generate: Сколько семплов генерировать для каждой модели
        n_bootstrap: Количество бутстрап-итераций
        confidence_level: Уровень доверительного интервала
        seed: Сид для воспроизводимости
        return_raw: Если True, возвращает (summary_df, raw_df)
    
    Returns:
        DataFrame со статистиками метрик для каждой модели
    """
    import torch
    
    rng = np.random.default_rng(seed)
    
    metrics_cols = [
        'mmd',              # MMD (меньше = лучше)
        'diversity',        # Разнообразие (больше = лучше)
        'coverage',         # Покрытие (больше = лучше)
        'mse_recon',        # Ошибка реконструкции (меньше = лучше)
        'mae_recon',        # MAE реконструкции (меньше = лучше)
        'latent_ks_stat',   # KS-статистика латентного пространства (меньше = лучше)
        'latent_p_value'    # p-value нормальности (больше = лучше)
    ]
    
    raw_records = []
    
    logging.info(f"=== GENERATION METRICS COMPARISON ===")
    logging.info(f"Real test samples: {X_real_test.shape[0]}")
    logging.info(f"Generating {n_generate} samples per model")
    logging.info(f"Bootstrap iterations: {n_bootstrap}")
    
    for b in range(n_bootstrap):
        # Бутстрап реальных данных (выборка с возвращением)
        n_real = X_real_test.shape[0]
        idx_real = rng.choice(n_real, size=n_real, replace=True)
        X_real_bs = X_real_test[idx_real]
        
        for name, model in models.items():
            model.eval()
            
            # 1. Генерация новых данных
            with torch.no_grad():
                # Сэмплируем из N(0, I)
                z_sample = torch.randn(n_generate, model.encoder.fc_mu.out_features).to(device)
                X_gen = model.decoder(z_sample).cpu().numpy()
            
            # 2. Реконструкция бутстрап-выборки
            with torch.no_grad():
                X_tensor = torch.tensor(X_real_bs, dtype=torch.float32).to(device)
                X_recon, _, _ = model(X_tensor)
                X_recon_np = X_recon.cpu().numpy()
            
            # 3. Вычисление метрик
            mmd = calculate_mmd(X_real_bs, X_gen)
            diversity = calculate_diversity(X_gen)
            coverage = calculate_coverage(X_real_bs, X_gen)
            
            recon_errors = np.mean((X_real_bs - X_recon_np) ** 2, axis=(1, 2))
            recon_abs_errors = np.mean(np.abs(X_real_bs - X_recon_np), axis=(1, 2))
            
            # 4. Проверка нормальности латентного пространства
            latent_stats = calculate_latent_normality(model, X_real_bs, device)
            
            rec = {
                'model': name,
                'iteration': b,
                'mmd': mmd,
                'diversity': diversity,
                'coverage': coverage,
                'mse_recon': np.mean(recon_errors),
                'mae_recon': np.mean(recon_abs_errors),
                'latent_ks_stat': latent_stats['mean_ks_stat'],
                'latent_p_value': latent_stats['mean_p_value']
            }
            
            raw_records.append(rec)
            
            if b % 20 == 0:
                logging.info(f"  Bootstrap {b}/{n_bootstrap} | Model: {name} | MMD: {mmd:.4f}")
    
    raw_df = pd.DataFrame(raw_records)
    
    # Агрегация статистик
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
    
    logging.info(f"Raw results shape: {raw_df.shape}")
    logging.info(f"Summary statistics:")
    logging.info(f"{summary_df.to_string(float_format=lambda x: f'{x:.4f}')}")
    
    return (summary_df, raw_df) if return_raw else summary_df


def log_generation_report(summary_df: pd.DataFrame) -> None:
    """Красиво выводит таблицу метрик генерации."""
    metrics = ['mmd', 'diversity', 'coverage', 'mse_recon', 'mae_recon', 
               'latent_ks_stat', 'latent_p_value']
    
    for metric in metrics:
        logging.info(f"{'='*60}")
        logging.info(f"Metric: {metric.upper()}")
        logging.info(f"{'='*60}")
        
        sub_df = summary_df[[c for c in summary_df.columns if c[0] == metric]]
        sub_df.columns = [c[1] for c in sub_df.columns]
        
        table_str = sub_df.to_string(float_format=lambda x: f'{x:.4f}')
        for line in table_str.split('\n'):
            logging.info(f"  {line}")
        
        logging.info("")