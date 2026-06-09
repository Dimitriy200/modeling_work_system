import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class TimeSeriesDeepSSM(nn.Module):
    def __init__(self, feature_dim, latent_dim, state_dim=32):
        """
        feature_dim: Количество датчиков (26)
        latent_dim: Размерность случайного шума Z (4)
        state_dim: Размерность скрытого физического состояния матрицы A (32)
        """
        super(TimeSeriesDeepSSM, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        # 1. СЕТЬ ПЕРЕХОДА СОСТОЯНИЙ (Уравнение Физики)
        # Принимает скрытое состояние h_{t-1} и случайный стиль Z_t, выдает новое состояние h_t
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim)
        )
        
        # 2. СЕТЬ НАБЛЮДЕНИЙ (Уравнение Датчиков)
        # Переводит скрытое состояние износа h_t в дельты изменения 26 датчиков
        self.emission_net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, feature_dim)
        )
        
        # 3. ЭНКОДЕР (Анализатор предыстории)
        # Сжимает известные 5 шагов в параметры латентного пространства Z
        self.encoder_rnn = nn.LSTM(input_size=feature_dim, hidden_size=state_dim, batch_first=True)
        self.fc_mu = nn.Linear(state_dim, latent_dim)
        self.fc_log_var = nn.Linear(state_dim, latent_dim)
        
        # Слой для инициализации самого первого скрытого состояния h_0 из истории
        self.fc_init_state = nn.Linear(state_dim, state_dim)

    def encode(self, x_past):
        _, (h_n, _) = self.encoder_rnn(x_past)
        h_last = h_n[-1] 
        mu = self.fc_mu(h_last)
        log_var = self.fc_log_var(h_last)
        return mu, log_var, h_last

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_past, last_known_step):
        """
        Обучение на 1 шаг вперед (предсказание 6-го шага)
        """
        mu, log_var, h_last = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        
        # Инициализируем текущее физическое состояние системы h_t
        h_t = F.relu(self.fc_init_state(h_last))
        
        # Уравнение перехода: комбинируем физику состояния и случайный латентный шум
        transition_input = torch.cat([h_t, z], dim=-1)
        h_next = self.transition_net(transition_input)
        
        # Уравнение эмиссии: переводим скрытое состояние в дельту датчиков
        delta = self.emission_net(h_next)
        
        # Прогноз = Точка опоры + Дельта из пространства состояний
        y_pred = last_known_step.unsqueeze(1) + delta.unsqueeze(1)
        
        return y_pred, mu, log_var

    # ----------------------------------------------------
    # ИНФЕРЕНС: МОДЕЛИРОВАНИЕ ТРАЕКТОРИИ ПРОСТРАНСТВА СОСТОЯНИЙ
    # ----------------------------------------------------
    def inference(self, x_past, horizon=10, num_scenarios=5):
        self.eval()
        batch_size = x_past.size(0)
        scenarios = []
        past_len = int(x_past.size(1) / 2)
        
        with torch.no_grad():
            for s in range(num_scenarios):
                current_history = x_past.clone()
                generated_window = []
                
                # Записываем известную историю (шаги 1-5)
                for t in range(past_len):
                    generated_window.append(x_past[:, t].unsqueeze(1))
                
                # Извлекаем начальные параметры из истории
                mu, log_var, h_last = self.encode(x_past)
                
                # Инициализируем скрытое состояние системы
                h_t = F.relu(self.fc_init_state(h_last))
                
                # Шаг 6-10: Эволюция пространства состояний во времени
                for t in range(past_len, horizon):
                    last_step = current_history[:, -1]
                    
                    # На КАЖДОМ шаге сэмплируем микро-шум в латентном пространстве SSM
                    z_t = self.reparameterize(mu, log_var)
                    
                    # 1. Обновляем скрытое состояние износа (Уравнение перехода)
                    transition_input = torch.cat([h_t, z_t], dim=-1)
                    h_t = self.transition_net(transition_input)
                    
                    # 2. Проецируем износ на датчики (Уравнение наблюдений)
                    delta = self.emission_net(h_t)
                    y_next_pred = last_step.unsqueeze(1) + delta.unsqueeze(1)
                    
                    generated_window.append(y_next_pred)
                    
                    # Обновляем скользящую историю
                    current_history = torch.cat([current_history[:, 1:], y_next_pred], dim=1)
                    
                scenario_tensor = torch.cat(generated_window, dim=1)
                scenarios.append(scenario_tensor.cpu().numpy())
                
        return scenarios

    # Встроенный метод обучения
    def fit(self, x_train, last_steps_train, y_train, epochs=150, lr=0.001, tau=0.15, verbose_step=20):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        history = {'total_loss': [], 'mse_loss': [], 'kl_loss': [], 'kl_weight': []}
        
        logging.info(f"--- START SSM TRAINING ({epochs} epochs) ---")
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
                logging.info(f"EPOCH {epoch:03d} | Loss: {total_loss.item():.4f} | MSE: {mse_loss.item():.4f} | KLD: {kl_loss_constrained.item():.4f}")
        return history
