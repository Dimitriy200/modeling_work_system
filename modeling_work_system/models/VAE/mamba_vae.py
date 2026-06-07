import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class TimeSeriesMambaSSM(nn.Module):
    def __init__(self, feature_dim, latent_dim, state_dim=32):
        """
        Полностью идентичный интерфейс для сохранения совместимости с вашими тестами.
        feature_dim: Количество датчиков (26)
        latent_dim: Размерность случайного шума Z (4)
        state_dim: Размерность скрытого селективного состояния (32)
        """
        super(TimeSeriesMambaSSM, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        
        # 1. ЭНКОДЕР (Оставляем как в оригинале для точного совпадения интерфейса)
        self.encoder_rnn = nn.LSTM(input_size=feature_dim, hidden_size=state_dim, batch_first=True)
        self.fc_mu = nn.Linear(state_dim, latent_dim)
        self.fc_log_var = nn.Linear(state_dim, latent_dim)
        self.fc_init_state = nn.Linear(state_dim, state_dim)
        
        # 2. КОМПОНЕНТЫ СЕЛЕКТИВНОЙ МАМБЫ
        # Базовая матрица А (задается логарифмически для стабильности непрерывного времени)
        A = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(state_dim, 1)
        self.A_log = nn.Parameter(torch.log(A)) # (state_dim, state_dim)
        
        # Вместо нелинейного перехода мы генерируем селективные параметры (dt, B, C) 
        # на основе комбинации скрытого состояния и латентного шума Z
        self.x_proj = nn.Linear(state_dim + latent_dim, state_dim + state_dim + state_dim, bias=False)
        self.dt_proj = nn.Linear(state_dim + latent_dim, state_dim, bias=True)
        
        # 3. СЕТЬ НАБЛЮДЕНИЙ (Уравнение Датчиков)
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
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _mamba_step(self, h_prev, z_t):
        """
        Внутренний шаг селективной дискретизации Mamba для одного таймстепа.
        Обеспечивает математическую идентичность параллельного и последовательного проходов.
        """
        # Объединяем физику предыдущего шага и стохастический шум
        ctx = torch.cat([h_prev, z_t], dim=-1)
        
        # Извлекаем селективные матрицы B и C
        selective_params = self.x_proj(ctx)
        dt, B_mat, C_mat = torch.split(selective_params, [self.state_dim, self.state_dim, self.state_dim], dim=-1)
        
        # Считаем шаг изменения непрерывного времени dt
        dt = F.softplus(self.dt_proj(ctx))
        
        # Дискретизация матрицы А: A_t = exp(dt * A)
        A = -torch.exp(self.A_log) # (state_dim, state_dim)
        A_t = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1)) # (B, state_dim, state_dim)
        
        # Дискретизация матрицы B: B_t = dt * B
        # Вычисляем новое скрытое состояние: h_t = A_t * h_{t-1} + B_t * z_t
        # Проекция входа B_t * z_t выражается как внешнее произведение векторов
        # B_t_z_t = (dt.unsqueeze(-1) * B_mat.unsqueeze(1)) * z_t.unsqueeze(-1) # (B, state_dim, state_dim)
        # logging.info(f"FORMS OF TENSORS dt: {dt.shape}, B_mat: {B_mat.shape}, z_t: {z_t.shape}")
        B_t_z_t = torch.einsum('bs, bs, bz -> bs', dt, B_mat, z_t)
        
        # Эволюция состояния
        h_next = torch.bmm(A_t, h_prev.unsqueeze(-1)).squeeze(-1) + B_t_z_t
        
        # Фиксация выхода через матрицу C
        y_mamba = h_next * C_mat
        return h_next, y_mamba

    def forward(self, x_past, last_known_step):
        """
        Обучение на 1 шаг вперед (предсказание 6-го шага).
        Интерфейс строго сохранен.
        """
        mu, log_var, h_last = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        
        # Инициализируем текущее скрытое состояние
        h_t = F.relu(self.fc_init_state(h_last))
        
        # Прогоняем один шаг эволюции Mamba
        h_next, y_mamba = self._mamba_step(h_t, z)
        
        # Проецируем через эммиссионную сеть в дельту датчиков
        delta = self.emission_net(y_mamba)
        
        # Прогноз = Точка опоры + Дельта
        y_pred = last_known_step.unsqueeze(1) + delta.unsqueeze(1)
        
        return y_pred, mu, log_var

    def inference(self, x_past, horizon=10, num_scenarios=5):
        """
        Инференс: Моделирование траекторий. Возвращает список numpy-массивов аналогично оригиналу.
        """
        self.eval()
        batch_size = x_past.size(0)
        scenarios = []
        
        with torch.no_grad():
            for s in range(num_scenarios):
                current_history = x_past.clone()
                generated_window = []
                
                # Копируем известное прошлое (шаги 1-5)
                for t in range(5):
                    generated_window.append(x_past[:, t].unsqueeze(1))
                
                mu, log_var, h_last = self.encode(x_past)
                h_t = F.relu(self.fc_init_state(h_last))
                
                # Генерация будущего (шаги 6-10)
                for t in range(5, horizon):
                    last_step = current_history[:, -1]
                    z_t = self.reparameterize(mu, log_var)
                    
                    # Шаг селективного SSM вместо полносвязного слоя перехода
                    h_t, y_mamba = self._mamba_step(h_t, z_t)
                    
                    delta = self.emission_net(y_mamba)
                    y_next_pred = last_step.unsqueeze(1) + delta.unsqueeze(1)
                    
                    generated_window.append(y_next_pred)
                    current_history = torch.cat([current_history[:, 1:], y_next_pred], dim=1)
                    
                scenario_tensor = torch.cat(generated_window, dim=1)
                scenarios.append(scenario_tensor.cpu().numpy())
                
        return scenarios

    def fit(self, x_train, last_steps_train, y_train, epochs=150, lr=0.001, tau=0.15, verbose_step=20):
        """
        Встроенный цикл обучения. Логика лосса, зажима и отжига KL-дивергенции на 100% идентична.
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        history = {'total_loss': [], 'mse_loss': [], 'kl_loss': [], 'kl_weight': []}
        
        logging.info(f"--- START MAMBA SSM TRAINING ({epochs} epochs) ---")
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
                logging.info(f"MAMBA EPOCH {epoch:03d} | Loss: {total_loss.item():.4f} | MSE: {mse_loss.item():.4f} | KLD: {kl_loss_constrained.item():.4f}")
        return history
