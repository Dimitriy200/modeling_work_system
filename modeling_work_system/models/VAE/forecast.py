import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging
from typing import Optional


class ForecastingEncoder(nn.Module):
    """
    Encoder, который читает ТОЛЬКО контекст (первые k шагов последовательности).
    Сжимает контекст в параметры латентного распределения.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_layers, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x_context):
        """
        x_context: (batch_size, context_len, input_dim)
        """
        _, (h_n, _) = self.lstm(x_context)
        h_last = h_n[-1]  # Берем выход последнего слоя
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class ForecastingDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, 
                 forecast_len: int, n_layers=2):
        super().__init__()
        self.forecast_len = forecast_len  # Сколько шагов предсказывать
        
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, z):
        # z: (batch, latent_dim)
        z_expanded = self.fc_init(z).unsqueeze(1).repeat(1, self.forecast_len, 1)
        out, _ = self.lstm(z_expanded)
        return self.fc_out(out)  # (batch, forecast_len, output_dim)
    

class Forecasting_VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, 
                 seq_len=40, context_len=20, forecast_len=20, n_layers=2):
        super().__init__()
        
        assert context_len + forecast_len == seq_len
        
        self.context_len = context_len
        self.forecast_len = forecast_len
        
        self.encoder = ForecastingEncoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = ForecastingDecoder(latent_dim, hidden_dim, input_dim, 
                                          forecast_len, n_layers)
    
    def forward(self, x_full):
        # x_full: (batch, 40, features)
        
        # 1. Encoder видит только контекст
        x_context = x_full[:, :self.context_len, :]  # (batch, 20, features)
        mu, logvar = self.encoder(x_context)
        
        # 2. Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # 3. Decoder генерирует только будущее
        x_forecast = self.decoder(z)  # (batch, 20, features)
        
        return x_forecast, mu, logvar
    
    def loss_function(self, x_forecast, x_full, mu, logvar, beta=1.0):
        # Loss только по "будущей" части!
        x_future = x_full[:, self.context_len:, :]  # (batch, 20, features)
        
        recon_loss = F.mse_loss(x_forecast, x_future, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + beta * kld_loss, recon_loss, kld_loss

    def calculate_forecasting_error(self, model, data_seq, device, batch_size=64):
        """
        Считает ошибку прогноза только по "будущей" части окна.
        """
        model.eval()
        errors = []
        
        with torch.no_grad():
            for i in range(0, len(data_seq), batch_size):
                batch = torch.tensor(data_seq[i:i+batch_size], dtype=torch.float32).to(device)
                
                # Forward pass
                x_forecast, mu, logvar = model(batch)
                
                # Берём только "будущую" часть
                x_future = batch[:, model.context_len:, :]
                
                # MSE только по будущему
                mse = torch.mean((x_future - x_forecast) ** 2, dim=(1, 2))
                errors.extend(mse.cpu().numpy())
        
        return np.array(errors)