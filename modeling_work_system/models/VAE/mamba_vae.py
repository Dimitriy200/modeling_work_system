import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class TimeSeriesMambaSSM(nn.Module):
    def __init__(self, feature_dim, latent_dim, state_dim=32):
        super(TimeSeriesMambaSSM, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        self.encoder_rnn = nn.LSTM(input_size=feature_dim, hidden_size=state_dim, batch_first=True)
        self.fc_mu = nn.Linear(state_dim, latent_dim)
        self.fc_log_var = nn.Linear(state_dim, latent_dim)
        self.fc_init_state = nn.Linear(state_dim, state_dim)
        
        self.z_to_state = nn.Linear(latent_dim, state_dim, bias=False)
        
        # Базовая матрица переходов A
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(state_dim, 1)
        self.A_log = nn.Parameter(torch.log(A)) 
        
        # Проекция контекста
        self.x_proj = nn.Linear(state_dim + latent_dim, state_dim * 3, bias=False)
        self.dt_proj = nn.Linear(state_dim + latent_dim, state_dim, bias=True)
        nn.init.constant_(self.dt_proj.bias, -2.0)
        
        # ИСПРАВЛЕНИЕ: Эмиссионная сеть теперь один в один повторяет логику DeepSSM (LSTM)
        # Она должна принимать скрытое состояние износа и переводить в дельту 26 датчиков
        self.emission_net = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, feature_dim)
        )

    def encode(self, x_past):
        _, (h_n, _) = self.encoder_rnn(x_past)
        h_last = h_n[-1] 
        mu = self.fc_mu(h_last)
        log_var = self.fc_log_var(h_last)
        return mu, log_var, h_last

    def reparameterize(self, mu, log_var):
        log_var = torch.clamp(log_var, min=-10.0, max=3.0)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _mamba_step(self, h_prev, z_t):
        ctx = torch.cat([h_prev, z_t], dim=-1)
        
        selective_params = self.x_proj(ctx)
        dt_raw, B_mat, C_mat = torch.split(selective_params, [self.state_dim, self.state_dim, self.state_dim], dim=-1)
        
        dt = F.softplus(self.dt_proj(ctx) + dt_raw)
        dt = torch.clamp(dt, min=1e-3, max=0.5)
        
        A = -torch.exp(torch.clamp(self.A_log, min=-5.0, max=5.0)) 
        A_t = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1)) 
        
        z_projected = self.z_to_state(z_t)
        B_t_z_t = (dt.unsqueeze(-1) * B_mat.unsqueeze(1)) * z_projected.unsqueeze(-1)
        
        # Эволюция физики состояния
        h_next = torch.bmm(A_t, h_prev.unsqueeze(-1)).squeeze(-1) + B_t_z_t.sum(dim=-1)
        h_next = torch.clamp(h_next, min=-5.0, max=5.0)
        
        # На выходе получаем строго отфильтрованный вектор состояния
        y_mamba = h_next * torch.tanh(C_mat)
        return h_next, y_mamba

    def forward(self, x_past, last_known_step):
        mu, log_var, h_last = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        
        h_t = F.relu(self.fc_init_state(h_last))
        h_next, y_mamba = self._mamba_step(h_t, z)
        
        # Вычисляем дельту
        delta = self.emission_net(y_mamba)
        
        # Прогноз = Точка опоры + Дельта из пространства состояний (строго как в LSTM)
        y_pred = last_known_step.unsqueeze(1) + delta.unsqueeze(1)
        return y_pred, mu, log_var

    def inference(self, x_past, horizon=10, num_scenarios=5):
        """
        Математически выверенный инференс. Полностью синхронизирован с методом forward.
        """
        self.eval()
        scenarios = []
        
        with torch.no_grad():
            for s in range(num_scenarios):
                current_history = x_past.clone()
                generated_window = []
                
                # 1. Заполняем первые 5 известных шагов
                for t in range(5):
                    generated_window.append(x_past[:, t].unsqueeze(1))
                
                # 2. Инициализируем контекст из истории
                mu, log_var, h_last = self.encode(x_past)
                h_t = F.relu(self.fc_init_state(h_last))
                
                # 3. Генерация шагов 6-10 (и далее до горизонта)
                for t in range(5, horizon):
                    last_step = current_history[:, -1]
                    z_t = self.reparameterize(mu, log_var)
                    
                    # Получаем корректно отфильтрованный вектор выхода y_mamba
                    h_t, y_mamba = self._mamba_step(h_t, z_t)
                    
                    # ПЕРЕДАЕМ В EMISSION СТРОГО ВЕКТОР Y_MAMBA, А НЕ МАТРИЦУ H_T!
                    delta = self.emission_net(y_mamba)
                    
                    # Прогноз = Точка опоры + Выученное смещение
                    y_next_pred = last_step.unsqueeze(1) + delta.unsqueeze(1)
                    
                    # Мягкая страховка от вычислительных аномалий
                    if torch.isnan(y_next_pred).any() or torch.isinf(y_next_pred).any():
                        y_next_pred = torch.nan_to_num(y_next_pred, nan=0.0)
                        y_next_pred = torch.where(y_next_pred == 0.0, last_step.unsqueeze(1), y_next_pred)
                    
                    generated_window.append(y_next_pred)
                    
                    # Обновляем окно истории для следующей итерации авторегрессии
                    current_history = torch.cat([current_history[:, 1:], y_next_pred], dim=1)
                    
                scenario_tensor = torch.cat(generated_window, dim=1)
                scenarios.append(scenario_tensor.cpu().numpy())
                
        return scenarios

    def fit(self, x_train, last_steps_train, y_train, epochs=300, lr=0.001, tau=0.15, verbose_step=20):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        history = {'total_loss': [], 'mse_loss': [], 'kl_loss': [], 'kl_weight': []}
        
        logging.info(f"--- START STABLE SELECTIVE MAMBA TRAINING ({epochs} epochs) ---")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            y_pred, mu, log_var = self.forward(x_train, last_steps_train)
            mse_loss = F.mse_loss(y_pred, y_train, reduction='mean')
            
            kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss_constrained = torch.clamp(kl_elementwise.mean(dim=0), min=tau).sum()
            
            start_annealing = int(epochs * 0.4) 
            kl_weight = 0.0 if epoch < start_annealing else min(0.05, (epoch - start_annealing) / (epochs - start_annealing))
            
            total_loss = mse_loss + (kl_weight * kl_loss_constrained)
            total_loss.backward()
            
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            optimizer.step()
            
            history['total_loss'].append(total_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['kl_loss'].append(kl_loss_constrained.item())
            history['kl_weight'].append(kl_weight)
            
            if epoch % verbose_step == 0 or epoch == epochs - 1:
                logging.info(f"MAMBA EPOCH {epoch:03d} | Loss: {total_loss.item():.4f} | MSE: {mse_loss.item():.4f} | KLD: {kl_loss_constrained.item():.4f}")
                
        return history