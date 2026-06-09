import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformerVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, d_model=64, nhead=4, num_layers=2):
        super(TimeSeriesTransformerVAE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Проекция сырых признаков в пространство d_model (требование Трансформера)
        self.feature_projection = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # ----------------------------------------------------
        # 1. ТРАНСФОРМЕР-ЭНКОДЕР
        # ----------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Вычисление параметров латентного распределения на основе финального вектора
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_log_var = nn.Linear(d_model, latent_dim)
        
        # ----------------------------------------------------
        # 2. ДЕКОДЕР
        # ----------------------------------------------------
        # Переводит латентный вектор Z обратно в состояние d_model для инициализации генерации
        self.fc_z_to_decoder = nn.Linear(latent_dim, d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Проекция из d_model обратно в физические признаки датчиков
        self.fc_out = nn.Linear(d_model, feature_dim)

    def encode(self, x_past):
        # x_past: (batch_size, seq_len, feature_dim)
        x = self.feature_projection(x_past) # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        
        memory = self.transformer_encoder(x) # (batch, seq_len, d_model)
        # В качестве агрегированного вектора берем среднее по временной оси (pooling)
        pooled = memory.mean(dim=1) # (batch, d_model)
        
        mu = self.fc_mu(pooled)
        log_var = self.fc_log_var(pooled)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_past, last_known_step):
        mu, log_var = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        
        # Проецируем Z и смешиваем его с последним известным шагом
        z_projected = self.fc_z_to_decoder(z).unsqueeze(1) # (batch, 1, d_model)
        last_step_proj = self.feature_projection(last_known_step.unsqueeze(1)) # (batch, 1, d_model)
        
        # Подаем в декодер комбинацию скрытого стиля Z и предыстории
        dec_input = last_step_proj + z_projected
        dec_out = self.transformer_decoder(dec_input)
        
        delta = self.fc_out(dec_out) # (batch, 1, feature_dim)
        y_pred = last_known_step.unsqueeze(1) + delta
        
        return y_pred, mu, log_var

    # ----------------------------------------------------
    # ИТЕРАТИВНЫЙ ИНФЕРЕНС С ТРАНСФОРМЕРОМ
    # ----------------------------------------------------
    def inference(self, x_past, horizon=10, num_scenarios=5):
        self.eval()
        scenarios = []
        past_len = int(x_past.size(1) / 2) 
        
        with torch.no_grad():
            for s in range(num_scenarios):
                current_history = x_past.clone()
                generated_window = []
                
                # Заполняем первые 5 циклов известной историей
                for t in range(past_len):
                    generated_window.append(x_past[:, t].unsqueeze(1))
                
                # Пошаговая генерация будущего (6-10 шаги)
                for t in range(past_len, horizon):
                    last_step = current_history[:, -1]
                    
                    # Генерируем ровно один следующий шаг
                    y_next_pred, _, _ = self.forward(current_history, last_step)
                    generated_window.append(y_next_pred)
                    
                    # Сдвигаем временное окно для следующей итерации
                    current_history = torch.cat([current_history[:, 1:], y_next_pred], dim=1)
                    
                scenario_tensor = torch.cat(generated_window, dim=1)
                scenarios.append(scenario_tensor.cpu().numpy())
                
        return scenarios

    # Встроенный метод обучения (остается классическим)
    def fit(self, x_train, last_steps_train, y_train, epochs=150, lr=0.001, tau=0.15, verbose_step=20):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'total_loss': [], 'mse_loss': [], 'kl_loss': [], 'kl_weight': []}
        
        logging.info(f"--- START OF TRANSFORMER TRAINING ({epochs} EPOCHS) ---")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            y_pred, mu, log_var = self.forward(x_train, last_steps_train)
            mse_loss = F.mse_loss(y_pred, y_train, reduction='mean')
            
            kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss_constrained = torch.clamp(kl_elementwise.mean(dim=0), min=tau).sum()
            
            start_annealing = int(epochs * 0.4)
            kl_weight = 0.0 if epoch < start_annealing else min(1.0, (epoch - start_annealing) / (epochs - start_annealing))
            
            total_loss = mse_loss + (kl_weight * kl_loss_constrained)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
            
            history['total_loss'].append(total_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['kl_loss'].append(kl_loss_constrained.item())
            history['kl_weight'].append(kl_weight)
            
            if epoch % verbose_step == 0 or epoch == epochs - 1:
                logging.info(f"EPOCH {epoch:03d} | Loss: {total_loss.item():.4f} | MSE (Next Step): {mse_loss.item():.4f} | KLD: {kl_loss_constrained.item():.4f}")
        return history
