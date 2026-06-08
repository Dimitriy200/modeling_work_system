import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging
from typing import Optional, Dict, Tuple, Any


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
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_layers)
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
    

    