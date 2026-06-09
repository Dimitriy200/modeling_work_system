import numpy as np
import torch
import logging
from scipy import stats
from sklearn.metrics import r2_score

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculates the 95% confidence interval for a 1D numpy array.
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean - h, mean + h

def evaluate_model_noise_robustness_advanced(model, X_stressed_past, Y_clean_full, past_steps=10, horizon=20):
    """
    Evaluates model stability by calculating Mean, Std, and 95% Confidence Intervals
    for RMSE, MAE, and R2 metrics across denoising and forecasting zones.
    """
    model.eval()
    device = next(model.parameters()).device
    
    logging.info("")
    logging.info("=== STARTING ADVANCED SCIENTIFIC METRIC EVALUATION ===")
    
    # 1. Batch inference over the entire validation dataset
    x_input = torch.FloatTensor(X_stressed_past).to(device)
    with torch.no_grad():
        gen_scenarios = model.inference(x_past=x_input, horizon=horizon, num_scenarios=5)
        # Calculate the mathematical expectation of the forecast (mean across 5 scenarios)
        y_pred_mean = np.mean(np.array(gen_scenarios), axis=0)
        
    if hasattr(Y_clean_full, 'values'): 
        Y_clean_full = Y_clean_full.values
        
    # Split the evaluation into two functional zones
    pred_denoise = y_pred_mean[:, :past_steps, :]
    true_denoise = Y_clean_full[:, :past_steps, :]
    
    pred_forecast = y_pred_mean[:, past_steps:, :]
    true_forecast = Y_clean_full[:, past_steps:, :]
    
    num_windows = len(Y_clean_full)
    
    rmse_w_denoise, mae_w_denoise, r2_w_denoise = [], [], []
    rmse_w_forecast, mae_w_forecast, r2_w_forecast = [], [], []
    
    # Compute metrics individually for each sliding window N
    for i in range(num_windows):
        # Denoising / Reconstruction Zone (Steps 1 to past_steps)
        rmse_w_denoise.append(np.sqrt(np.mean((pred_denoise[i] - true_denoise[i]) ** 2)))
        mae_w_denoise.append(np.mean(np.abs(pred_denoise[i] - true_denoise[i])))
        r2_w_denoise.append(r2_score(true_denoise[i].flatten(), pred_denoise[i].flatten()))
        
        # Forecasting Zone (Steps past_steps+1 to horizon)
        rmse_w_forecast.append(np.sqrt(np.mean((pred_forecast[i] - true_forecast[i]) ** 2)))
        mae_w_forecast.append(np.mean(np.abs(pred_forecast[i] - true_forecast[i])))
        r2_w_forecast.append(r2_score(true_forecast[i].flatten(), pred_forecast[i].flatten()))
        
    metrics = {
        f"Reconstruction Zone (Steps 1-{past_steps})": {
            "RMSE": np.array(rmse_w_denoise), "MAE": np.array(mae_w_denoise), "R2": np.array(r2_w_denoise)
        },
        f"Forecasting Zone (Steps {past_steps+1}-{horizon})": {
            "RMSE": np.array(rmse_w_forecast), "MAE": np.array(mae_w_forecast), "R2": np.array(r2_w_forecast)
        }
    }
    
    # 2. Structured logging of distribution statistics
    for zone_name, zone_metrics in metrics.items():
        logging.info(f"--- Functional Zone: {zone_name} ---")
        for metric_name, dist in zone_metrics.items():
            mean = np.mean(dist)
            std = np.std(dist)  # Standard deviation
            ci_low, ci_high = calculate_confidence_interval(dist, confidence=0.95)
            
            logging.info(
                f"{metric_name:4} -> Mean: {mean:.4f} | Std: {std:.4f} | 95% CI: [{ci_low:.4f}; {ci_high:.4f}]"
            )
            
    # Returns the window-wise forecasting RMSE vector for Student's t-test validation
    return metrics[f"Forecasting Zone (Steps {past_steps+1}-{horizon})"]["RMSE"]
