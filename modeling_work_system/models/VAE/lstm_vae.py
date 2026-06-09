import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging



# Пересмотреть дектодер - нужен слой lstm

class TimeSeriesIterativeVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dim=64):
        super(TimeSeriesIterativeVAE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # ЭНКОДЕР (Анализирует окно истории)
        self.encoder_lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # ДЕКОДЕР (Генерирует строго 1 следующий шаг)
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim // 2)
        # Вход декодера: латентный вектор Z. Мы убираем LSTM из декодера, оставляя GRU или Dense, 
        # так как шаг всего один, и последовательность мы строим внешним циклом.
        self.decoder_dense = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, feature_dim)
        )

    def encode(self, x_past):
        _, (h_n, _) = self.encoder_lstm(x_past)
        h_last = h_n[-1] 
        mu = self.fc_mu(h_last)
        log_var = self.fc_log_var(h_last)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_past, last_known_step):
        mu, log_var = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        
        # Предсказываем дельту только для ОДНОГО следующего шага
        delta = self.decoder_dense(z) 
        
        # Следующий шаг = Последний известный + Дельта
        y_pred = last_known_step + delta
        return y_pred.unsqueeze(1), mu, log_var # возвращаем (batch, 1, feature_dim) для совместимости

    # ----------------------------------------------------
    # ИТЕРАТИВНЫЙ ИНФЕРЕНС (Генерация цепочки по одному шагу)
    # ----------------------------------------------------
    def inference(self, x_past, horizon=10, num_scenarios=5):
        """
        x_past: Стартовые 5 циклов истории формы (batch_size, 5, feature_dim)
        horizon: Полная длина генерируемого окна (10 шагов)
        """
        self.eval()
        device = x_past.device
        batch_size = x_past.size(0)
        
        # Список, где мы будем копить сценарии. 
        # Каждый сценарий будет массивом формы (batch_size, 10, feature_dim)
        scenarios = []
        
        with torch.no_grad():
            for s in range(num_scenarios):
                # Создаем копию стартовой истории для текущего сценария
                current_history = x_past.clone() 
                
                # Буфер для хранения всех 10 шагов этого сценария
                generated_window = []
                
                # Шаг 1-5: Сначала просто копируем реальную предысторию в наш выходной буфер
                for t in range(5):
                    generated_window.append(x_past[:, t].unsqueeze(1))
                
                # Шаг 6-10: Начинаем итеративную генерацию будущего
                for t in range(5, horizon):
                    # Точка опоры — это всегда самый последний шаг в текущей истории (индекс -1)
                    last_step = current_history[:, -1]
                    
                    # Прогоняем текущую историю через VAE, чтобы получить ОДИН следующий шаг
                    y_next_pred, _, _ = self.forward(current_history, last_step) # (batch, 1, feature_dim)
                    
                    # Сохраняем предсказанный шаг в буфер будущего
                    generated_window.append(y_next_pred)
                    
                    # ОБНОВЛЯЕМ ИСТОРИЮ: выкидываем самый старый шаг (индекс 0) 
                    # и добавляем только что предсказанный шаг в конец последовательности
                    current_history = torch.cat([current_history[:, 1:], y_next_pred], dim=1)
                
                # Объединяем все 10 шагов в единый тензор сценария
                scenario_tensor = torch.cat(generated_window, dim=1) # (batch, 10, feature_dim)
                scenarios.append(scenario_tensor.cpu().numpy())
                
        return scenarios

    # Обновленный метод fit под 1 шаг предсказания
    def fit(self, x_train, last_steps_train, y_train, epochs=150, lr=0.001, tau=0.15, verbose_step=20):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'total_loss': [], 'mse_loss': [], 'kl_loss': [], 'kl_weight': []}
        
        logging.info(f"--- START OF AUTOREGRESSIVE LEARNING ({epochs} EPOCHS) ---")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            y_pred, mu, log_var = self.forward(x_train, last_steps_train)
            
            # Лосс считаем строго по ОДНОМУ следующему шагу (шагу 6)
            mse_loss = F.mse_loss(y_pred, y_train, reduction='mean')
            
            kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss_constrained = torch.clamp(kl_elementwise.mean(dim=0), min=tau).sum()
            
            # start_annealing = int(epochs * 0.4)
            # kl_weight = 0.0 if epoch < start_annealing else min(0.05, (epoch - start_annealing) / (epochs - start_annealing))
            start_annealing = int(epochs * 0.4) # 60

            if epoch < start_annealing:
                kl_weight = 0.0
            else:
                # Делим на 60, чтобы вес рос ровно 60 эпох
                growth_progress = (epoch - start_annealing) / 60.0 
                
                # Плавно увеличиваем вес от 0.0 до 0.05
                kl_weight = min(0.05, growth_progress * 0.05) 
            
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
