"""
Adaptive Forecasting VAE для детекции аномалий.
Архитектура:
Encoder: LSTM → сжимает контекст (первые k шагов) в латентное пространство.
Decoder: Autoregressive LSTM → генерирует полную последовательность шаг за шагом.
Loss: Взвешенная сумма ошибок контекста и будущего, где веса α и β
являются обучаемыми параметрами (Learnable Softmax).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging
from typing import Optional, Tuple

# ======================================================
# 1. ENCODER (Читает только контекст)
# ======================================================
class ForecastingEncoder(nn.Module):
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

    def forward(self, x_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_context: (batch, context_len, input_dim)
        """
        _, (h_n, _) = self.lstm(x_context)
        # logging.info(f"_, (h_n, _) = {_, (h_n, _)}")

        h_last = h_n[-1]
        # logging.info(f"h_last = {h_last}")
        
        mu = self.fc_mu(h_last)
        # logging.info(f"mu = {mu}")

        logvar = self.fc_logvar(h_last)
        # logging.info(f"logvar = {logvar}")

        return mu, logvar

# ======================================================
# 2. DECODER (Autoregressive - генерирует шаг за шагом)
# ======================================================
class AutoregressiveDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 seq_len: int, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Проецируем z в начальное скрытое состояние
        self.fc_h0 = nn.Linear(latent_dim, hidden_dim * n_layers)
        self.fc_c0 = nn.Linear(latent_dim, hidden_dim * n_layers)
        
        # ✅ LSTM принимает конкатенацию [x, z]
        lstm_input_dim = output_dim + latent_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim, n_layers, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, latent_dim)
        x_context: (batch, context_len, output_dim) - для teacher forcing
        
        Возвращает: (batch, seq_len, output_dim)
        """
        batch_size = z.size(0)
        
        # Инициализируем скрытые состояния из z
        h0 = self.fc_h0(z).view(self.n_layers, batch_size, self.hidden_dim)
        c0 = self.fc_c0(z).view(self.n_layers, batch_size, self.hidden_dim)
        
        # Создаем входную последовательность: контекст + нули для будущего
        x_input = torch.zeros(batch_size, self.seq_len, self.output_dim).to(z.device)
        x_input[:, :x_context.size(1), :] = x_context  # Первые шаги = контекст
        
        # Дублируем z на всю длину seq_len
        z_expanded = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # ✅ Конкатенируем x и z
        lstm_input = torch.cat([x_input, z_expanded], dim=-1)
        
        # LSTM генерирует последовательность
        out, _ = self.lstm(lstm_input, (h0, c0))
        
        # Финальная проекция
        return self.fc_out(out)

# ======================================================
# 3. MAIN MODEL (Adaptive Forecasting VAE)
# ======================================================
class AdaptiveForecasting_VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 seq_len: int, context_len: int, forecast_len: int,
                 n_layers: int = 2,
                 init_alpha: float = 0.5, init_beta: float = 0.5):
        super().__init__()
        
        if context_len + forecast_len != seq_len:
            raise ValueError("context_len + forecast_len должно равняться seq_len")
        
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Обучаемые веса
        log_alpha = np.log(init_alpha)
        log_beta = np.log(init_beta)
        
        self.log_weights = nn.Parameter(
            torch.tensor([log_alpha, log_beta], dtype=torch.float32)
        )
        
        # Архитектура
        self.encoder = ForecastingEncoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = AutoregressiveDecoder(latent_dim, hidden_dim, input_dim, seq_len, n_layers)

    def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.log_weights, dim=0)
        MIN_WEIGHT = 0.1  # Не даем весам упасть ниже 10%
        weights = torch.clamp(weights, min=MIN_WEIGHT)
        weights = weights / weights.sum()  # Нормализуем обратно
        return weights[0], weights[1]

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_context = x_full[:, :self.context_len, :]
        mu, logvar = self.encoder(x_context)
        z = self.reparameterize(mu, logvar)
        x_recon_full = self.decoder(z, x_context)
        
        return x_recon_full, mu, logvar

    def loss_function(
            self, 
            x_recon_full: torch.Tensor, 
            x_full: torch.Tensor, 
            mu: torch.Tensor, 
            logvar: torch.Tensor, 
            beta_kl: float = 1.0):
        
        alpha, beta = self.get_weights()
        
        x_recon_context = x_recon_full[:, :self.context_len, :]
        x_recon_forecast = x_recon_full[:, self.context_len:, :]
        
        x_context = x_full[:, :self.context_len, :]
        x_future = x_full[:, self.context_len:, :]
        
        loss_context = F.mse_loss(x_recon_context, x_context, reduction='mean')
        loss_forecast = F.mse_loss(x_recon_forecast, x_future, reduction='mean')
        
        # KL Divergence
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        FREE_BITS = 1.0  # Минимальный KL, который модель обязана использовать
        # total_loss = alpha * loss_context + beta * loss_forecast + FREE_BITS * beta_kl * kld_loss
        total_loss = alpha * loss_context + beta * loss_forecast + beta_kl * kld_loss
        # kld_loss = torch.max(kld_loss, torch.tensor(FREE_BITS).to(kld_loss.device))
        
        return total_loss, loss_context, loss_forecast, kld_loss, alpha, beta

    def generate(
            self, 
            x_context: torch.Tensor, 
            n_samples: int = 1, 
            temperature: float = 1.0) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        x_context = x_context.to(device)
        batch_size = x_context.size(0)
        
        generated_sequences = []
        with torch.no_grad():
            mu, logvar = self.encoder(x_context)
            for _ in range(n_samples):
                std = torch.exp(0.5 * logvar) * temperature
                z = mu + torch.randn_like(std)
                
                # Инициализация скрытых состояний
                h0 = self.decoder.fc_h0(z).view(self.decoder.n_layers, batch_size, self.decoder.hidden_dim)
                c0 = self.decoder.fc_c0(z).view(self.decoder.n_layers, batch_size, self.decoder.hidden_dim)
                hidden = (h0, c0)
                
                x_gen = torch.zeros(batch_size, self.seq_len, x_context.size(2)).to(device)
                x_gen[:, :self.context_len, :] = x_context
                
                for t in range(self.seq_len):
                    if t < self.context_len:
                        current_input = x_gen[:, t:t+1, :]
                    else:
                        current_input = x_gen[:, t-1:t, :]
                    
                    # Добавляем z к входу на каждом шаге
                    z_step = z.unsqueeze(1)
                    lstm_in = torch.cat([current_input, z_step], dim=-1)
                    
                    out, hidden = self.decoder.lstm(lstm_in, hidden)
                    x_gen[:, t:t+1, :] = self.decoder.fc_out(out)
                    
                generated_sequences.append(x_gen.cpu().numpy())
        
        result = np.stack(generated_sequences, axis=0)
        
        logging.info(f"Generated shape: {result.shape}")  # Для отладки
        return result

    def generate_from_noise(self, n_samples: int = 100, 
                           temperature: float = 1.0) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device) * temperature
            
            # Генерируем из шума (без контекста)
            x_gen = torch.zeros(n_samples, self.seq_len, self.decoder.fc_out.out_features).to(device)
            
            h0 = self.decoder.fc_h0(z).view(self.decoder.n_layers, n_samples, self.decoder.hidden_dim)
            c0 = self.decoder.fc_c0(z).view(self.decoder.n_layers, n_samples, self.decoder.hidden_dim)
            
            hidden = (h0, c0)
            for t in range(self.seq_len):
                if t == 0:
                    lstm_input = torch.zeros(n_samples, 1, self.decoder.fc_out.out_features).to(device)
                else:
                    lstm_input = x_gen[:, t-1:t, :]
                
                out, hidden = self.decoder.lstm(lstm_input, hidden)
                x_gen[:, t:t+1, :] = self.decoder.fc_out(out)
            
            return x_gen.cpu().numpy()

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, 
            epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, 
            warmup_epochs: int = 10, device: str = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)), 
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32)), 
            batch_size=batch_size, shuffle=False
        )
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'train_loss': [], 'val_loss': [], 
            'train_context': [], 'train_forecast': [], 'train_kld': [],
            'alpha_history': [], 'beta_history': []
        }
        
        logging.info(f"=== STARTING ADAPTIVE FORECASTING VAE ===")
        logging.info(f"Initial α={self.get_weights()[0].item():.3f}, β={self.get_weights()[1].item():.3f}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            beta_kl = min(1.0, epoch / warmup_epochs)
            
            # === TRAIN ===
            self.train()
            epoch_train_loss = epoch_context = epoch_forecast = epoch_kld = 0.0
            
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                
                x_recon, mu, logvar = self(x)
                loss, lc, lf, kl, alpha, beta = self.loss_function(x_recon, x, mu, logvar, beta_kl)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_context += lc.item()
                epoch_forecast += lf.item()
                epoch_kld += kl.item()
            
            alpha_val, beta_val = self.get_weights()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_context = epoch_context / len(train_loader)
            avg_forecast = epoch_forecast / len(train_loader)
            avg_kld = epoch_kld / len(train_loader)
            
            # === VAL ===
            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_recon, mu, logvar = self(x)
                    loss, _, _, _, _, _ = self.loss_function(x_recon, x, mu, logvar, beta_kl)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
            else:
                patience_counter += 1
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_context'].append(avg_context)
            history['train_forecast'].append(avg_forecast)
            history['train_kld'].append(avg_kld)
            history['alpha_history'].append(alpha_val.item())
            history['beta_history'].append(beta_val.item())
            
            if epoch % 5 == 0 or epoch == 1:
                logging.info(
                    f"Epoch {epoch:03d} | α={alpha_val.item():.3f} β={beta_val.item():.3f} | "
                    f"Total: {avg_train_loss:.4f} | Ctx: {avg_context:.4f} | "
                    f"Fcst: {avg_forecast:.4f} | KLD: {avg_kld:.4f} | Val: {avg_val_loss:.4f}"
                )
            
            if patience_counter >= 10:
                logging.info(f"Early stopping at epoch {epoch}")
                self.load_state_dict(best_state)
                break
        
        logging.info("=== TRAINING FINISHED ===")
        return history







# """
# Adaptive Forecasting VAE для детекции аномалий.
# Архитектура:
#   - Encoder: LSTM → сжимает контекст (первые k шагов) в латентное пространство.
#   - Decoder: LSTM → генерирует полную последовательность (все N шагов).
#   - Loss: Взвешенная сумма ошибок контекста и будущего, где веса α и β 
#           являются обучаемыми параметрами (Learnable Softmax).
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.utils.data import TensorDataset, DataLoader
# import logging
# from typing import Optional, Tuple


# # ======================================================
# # 1. ENCODER (Читает только контекст)
# # ======================================================
# class ForecastingEncoder(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
        
#         self.lstm = nn.LSTM(
#             input_dim, hidden_dim, n_layers, 
#             batch_first=True, 
#             dropout=0.2 if n_layers > 1 else 0
#         )
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

#     def forward(self, x_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         x_context: (batch, context_len, input_dim)
#         """
#         _, (h_n, _) = self.lstm(x_context)
#         h_last = h_n[-1]  # Берем скрытое состояние последнего слоя
        
#         mu = self.fc_mu(h_last)
#         logvar = self.fc_logvar(h_last)
#         return mu, logvar


# # ======================================================
# # 2. DECODER (Генерирует полную последовательность)
# # ======================================================
# class FullSequenceDecoder(nn.Module):
#     def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, 
#                  seq_len: int, n_layers: int = 2):
#         super().__init__()
#         self.seq_len = seq_len
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
        
#         self.fc_init = nn.Linear(latent_dim, hidden_dim)
#         self.lstm = nn.LSTM(
#             hidden_dim, hidden_dim, n_layers, 
#             batch_first=True, 
#             dropout=0.2 if n_layers > 1 else 0
#         )
#         self.fc_out = nn.Linear(hidden_dim, output_dim)

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         z: (batch, latent_dim)
#         Возвращает: (batch, seq_len, output_dim)
#         """
#         # Проецируем z и дублируем на всю длину seq_len
#         z_expanded = self.fc_init(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        
#         # LSTM генерирует временную динамику
#         out, _ = self.lstm(z_expanded)
        
#         # Финальная проекция в пространство признаков
#         return self.fc_out(out)


# # ======================================================
# # 3. MAIN MODEL (Adaptive Forecasting VAE)
# # ======================================================
# class AdaptiveForecasting_VAE(nn.Module):
#     """
#     VAE с адаптивными весами loss через Learnable Softmax.
#     """
#     def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
#                  seq_len: int, context_len: int, forecast_len: int, 
#                  n_layers: int = 2,
#                  init_alpha: float = 0.5, init_beta: float = 0.5):
#         super().__init__()
        
#         if context_len + forecast_len != seq_len:
#             raise ValueError("context_len + forecast_len должно равняться seq_len")
        
#         self.context_len = context_len
#         self.forecast_len = forecast_len
#         self.seq_len = seq_len
#         self.latent_dim = latent_dim
        
#         # ======================================================
#         # ОБУЧАЕМЫЕ ВЕСА (Ключевая особенность)
#         # ======================================================
#         # Инициализируем в логарифмическом пространстве для стабильности softmax
#         log_alpha = np.log(init_alpha)
#         log_beta = np.log(init_beta)
        
#         self.log_weights = nn.Parameter(
#             torch.tensor([log_alpha, log_beta], dtype=torch.float32)
#         )
        
#         # Архитектура
#         self.encoder = ForecastingEncoder(input_dim, hidden_dim, latent_dim, n_layers)
#         self.decoder = FullSequenceDecoder(latent_dim, hidden_dim, input_dim, seq_len, n_layers)

#     def get_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Вычисляет текущие веса α и β через softmax."""
#         weights = F.softmax(self.log_weights, dim=0)
#         return weights[0], weights[1]

#     def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
#         """Трюк репараметризации."""
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x_full: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#         """
#         x_full: (batch, seq_len, input_dim)
#         """
#         # 1. Encoder видит только контекст
#         x_context = x_full[:, :self.context_len, :]
#         mu, logvar = self.encoder(x_context)
        
#         # 2. Сэмплируем латентный вектор
#         z = self.reparameterize(mu, logvar)
        
#         # 3. Decoder генерирует ВСЮ последовательность
#         x_recon_full = self.decoder(z)
        
#         return x_recon_full, mu, logvar

#     def loss_function(self, x_recon_full: torch.Tensor, x_full: torch.Tensor, 
#                      mu: torch.Tensor, logvar: torch.Tensor, beta_kl: float = 1.0):
#         """
#         Взвешенная функция потерь с динамическими весами.
#         """
#         # Получаем текущие веса (они же градиентно обновляются!)
#         # alpha, beta = self.get_weights()
#         alpha = 0.3  # Контекст
#         beta = 0.7   # Будущее

        
#         # Разделяем на контекст и будущее
#         x_recon_context = x_recon_full[:, :self.context_len, :]
#         x_recon_forecast = x_recon_full[:, self.context_len:, :]
        
#         x_context = x_full[:, :self.context_len, :]
#         x_future = x_full[:, self.context_len:, :]
        
#         # Ошибки реконструкции (используем sum для баланса с KL)
#         loss_context = F.mse_loss(x_recon_context, x_context, reduction='mean')
#         loss_forecast = F.mse_loss(x_recon_forecast, x_future, reduction='mean')
        
#         # KL Divergence
#         kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
#         # Итоговый loss
#         # total_loss = alpha * loss_context + beta * loss_forecast + beta_kl * kld_loss
#         total_loss = alpha * loss_context + beta * loss_forecast + 0.01 * beta_kl * kld_loss
        
#         return total_loss, loss_context, loss_forecast, kld_loss, alpha, beta

#     # ======================================================
#     # МЕТОДЫ ГЕНЕРАЦИИ (Для инференса)
#     # ======================================================
#     def generate(self, x_context: torch.Tensor, n_samples: int = 1, 
#                 temperature: float = 1.0) -> np.ndarray:
#         """Генерация полной последовательности по контексту."""
#         self.eval()
#         device = next(self.parameters()).device
#         x_context = x_context.to(device)
        
#         generated_sequences = []
#         with torch.no_grad():
#             mu, logvar = self.encoder(x_context)
#             for _ in range(n_samples):
#                 std = torch.exp(0.5 * logvar) * temperature
#                 eps = torch.randn_like(std)
#                 z = mu + eps * std
#                 x_gen = self.decoder(z)
#                 generated_sequences.append(x_gen.cpu().numpy())
#         return np.array(generated_sequences)

#     def generate_from_noise(self, n_samples: int = 100, 
#                            temperature: float = 1.0) -> np.ndarray:
#         """Генерация из чистого шума."""
#         self.eval()
#         device = next(self.parameters()).device
#         with torch.no_grad():
#             z = torch.randn(n_samples, self.latent_dim).to(device) * temperature
#             x_gen = self.decoder(z)
#         return x_gen.cpu().numpy()

#     # ======================================================
#     # МЕТОД ОБУЧЕНИЯ (FIT)
#     # ======================================================
#     def fit(self, X_train: np.ndarray, X_val: np.ndarray, 
#             epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, 
#             warmup_epochs: int = 10, device: str = None):
#         """Обучение модели с логированием динамики весов."""
#         if device is None:
#             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.to(device)
        
#         train_loader = DataLoader(
#             TensorDataset(torch.tensor(X_train, dtype=torch.float32)), 
#             batch_size=batch_size, shuffle=True, drop_last=True
#         )
#         val_loader = DataLoader(
#             TensorDataset(torch.tensor(X_val, dtype=torch.float32)), 
#             batch_size=batch_size, shuffle=False
#         )
        
#         # Adam оптимизирует ВСЕ параметры, включая log_weights!
#         optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-3)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=5
#         )
        
#         history = {
#             'train_loss': [], 'val_loss': [], 
#             'train_context': [], 'train_forecast': [], 'train_kld': [],
#             'alpha_history': [], 'beta_history': []
#         }
        
#         logging.info(f"=== STARTING ADAPTIVE FORECASTING VAE ===")
#         logging.info(f"Initial α={self.get_weights()[0].item():.3f}, β={self.get_weights()[1].item():.3f}")
        
#         best_val_loss = float('inf')
#         patience_counter = 0
        
#         for epoch in range(1, epochs + 1):
#             beta_kl = min(1.0, epoch / warmup_epochs)
            
#             # === TRAIN ===
#             self.train()
#             epoch_train_loss = epoch_context = epoch_forecast = epoch_kld = 0.0
            
#             for batch in train_loader:
#                 x = batch[0].to(device)
#                 optimizer.zero_grad()
                
#                 x_recon, mu, logvar = self(x)
#                 loss, lc, lf, kl, alpha, beta = self.loss_function(x_recon, x, mu, logvar, beta_kl)
                
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
#                 optimizer.step()
                
#                 epoch_train_loss += loss.item()
#                 epoch_context += lc.item()
#                 epoch_forecast += lf.item()
#                 epoch_kld += kl.item()
            
#             # Сохраняем веса в историю
#             alpha_val, beta_val = self.get_weights()
            
#             avg_train_loss = epoch_train_loss / len(train_loader)
#             avg_context = epoch_context / len(train_loader)
#             avg_forecast = epoch_forecast / len(train_loader)
#             avg_kld = epoch_kld / len(train_loader)
            
#             # === VAL ===
#             self.eval()
#             epoch_val_loss = 0.0
#             with torch.no_grad():
#                 for batch in val_loader:
#                     x = batch[0].to(device)
#                     x_recon, mu, logvar = self(x)
#                     loss, _, _, _, _, _ = self.loss_function(x_recon, x, mu, logvar, beta_kl)
#                     epoch_val_loss += loss.item()
            
#             avg_val_loss = epoch_val_loss / len(val_loader)
#             scheduler.step(avg_val_loss)
            
#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#                 best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
#             else:
#                 patience_counter += 1
            
#             # Заполняем историю
#             history['train_loss'].append(avg_train_loss)
#             history['val_loss'].append(avg_val_loss)
#             history['train_context'].append(avg_context)
#             history['train_forecast'].append(avg_forecast)
#             history['train_kld'].append(avg_kld)
#             history['alpha_history'].append(alpha_val.item())
#             history['beta_history'].append(beta_val.item())
            
#             if epoch % 5 == 0 or epoch == 1:
#                 logging.info(
#                     f"Epoch {epoch:03d} | α={alpha_val.item():.3f} β={beta_val.item():.3f} | "
#                     f"Total: {avg_train_loss:.4f} | Ctx: {avg_context:.4f} | "
#                     f"Fcst: {avg_forecast:.4f} | KLD: {avg_kld:.4f} | Val: {avg_val_loss:.4f}"
#                 )
            
#             if patience_counter >= 10:
#                 logging.info(f"Early stopping at epoch {epoch}")
#                 self.load_state_dict(best_state)
#                 break
        
#         logging.info("=== TRAINING FINISHED ===")
#         return history