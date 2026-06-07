import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class TimeSeriesForecastingVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dim=64):
        super(TimeSeriesForecastingVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # ----------------------------------------------------
        # 1. ЭНКОДЕР (Анализирует прошлое)
        # ----------------------------------------------------
        self.encoder_lstm = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Слой для вычисления Среднего (Mean)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        # Слой для вычисления Логарифма дисперсии (Log-Variance)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # ----------------------------------------------------
        # 2. ДЕКОДЕР (Генерирует будущее)
        # ----------------------------------------------------
        # Вход декодера: латентный вектор Z. Мы сделаем декодер чуть проще энкодера,
        # чтобы он не доминировал и не игнорировал скрытое пространство.
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim // 2)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim // 2, 
            hidden_size=hidden_dim // 2, 
            num_layers=1, 
            batch_first=True
        )
        
        # Финальный слой выдает дельту изменения для следующего шага
        self.fc_out = nn.Linear(hidden_dim // 2, feature_dim)

    def encode(self, x_past):
        # x_past: (batch, seq_len, feature_dim)
        _, (h_n, _) = self.encoder_lstm(x_past)
        # h_n имеет форму (1, batch, hidden_dim). Убираем первую размерность:
        h_last = h_n[-1] 
        
        mu = self.fc_mu(h_last)
        log_var = self.fc_log_var(h_last)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        # Трюк репараметризации
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, horizon=1):
        # horizon: на сколько шагов вперед предсказываем (в нашем случае на 1 шаг)
        batch_size = z.size(0)
        
        # Преобразуем латентный вектор в начальное состояние для декодирования
        dec_input = F.relu(self.decoder_fc(z)) # (batch, hidden_dim//2)
        
        # Повторяем вектор для нужного горизонта прогнозирования
        dec_input = dec_input.unsqueeze(1).repeat(1, horizon, 1) # (batch, horizon, hidden_dim//2)
        
        out, _ = self.decoder_lstm(dec_input) # (batch, horizon, hidden_dim//2)
        delta = self.fc_out(out) # (batch, horizon, feature_dim)
        return delta

    def forward(self, x_past, last_known_step, horizon=1):
        mu, log_var = self.encode(x_past)
        z = self.reparameterize(mu, log_var)
        
        delta = self.decode(z, horizon=horizon)
        
        # RESIDUAL ТРЮК: Будущее = Последний известный шаг + Предсказанное изменение (дельта)
        # Нам нужно расширить размерность last_known_step до (batch, horizon, feature_dim)
        y_pred = last_known_step.unsqueeze(1) + delta
        
        return y_pred, mu, log_var

    def fit(self, X_train, last_steps_train, y_train, epochs=500, lr=0.005, tau=0.15, verbose_step=100):
        """
        Метод для обучения модели на переданных тензорах.
        Автоматически рассчитывает KL Annealing, Free Bits и делает шаг оптимизации.
        """
        self.train() # Переводим модель в режим обучения
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        logging.info(f"=== START MODEL TRAINING ({epochs} epochs) ===")

        history = {
            'total_loss': [],
            'mse_loss': [],
            'kl_loss': [],
            'kl_weight': []
        }
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Прямой проход (подразумеваем горизонт прогнозирования равным длине y_train)
            horizon = y_train.size(1)
            y_pred, mu, log_var = self.forward(X_train, last_steps_train, horizon=horizon)
            
            # 1. Расчет ошибки прогноза (MSE)
            mse_loss = F.mse_loss(y_pred, y_train, reduction='mean')
            
            # 2. Расчет KLD с Free Bits (Свободные биты по размерностям)
            kl_elementwise = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
            kl_loss_per_dim = kl_elementwise.mean(dim=0)
            kl_loss_constrained = torch.clamp(kl_loss_per_dim, min=tau).sum()
            
            # 3. Расчет KL Annealing веса (включение после 25% эпох)
            start_annealing = int(epochs * 0.25)
            if epoch < start_annealing:
                kl_weight = 0.0
            else:
                kl_weight = min(1.0, (epoch - start_annealing) / (epochs - start_annealing))
                
            # Итоговая функция потерь
            total_loss = mse_loss + (kl_weight * kl_loss_constrained)
            
            # Шаг обратного распространения ошибки
            total_loss.backward()
            
            # Градиентный клиппинг (защита от взрыва градиентов при скачках KLD)
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            optimizer.step()

            history['total_loss'].append(total_loss.item())
            history['mse_loss'].append(mse_loss.item())
            history['kl_loss'].append(kl_loss_constrained.item())
            history['kl_weight'].append(kl_weight)
            
            # Логирование процесса
            if epoch % verbose_step == 0 or epoch == epochs - 1:
                logging.info(f"EPOCH {epoch:03d} | Total Loss: {total_loss.item():.4f} | MSE: {mse_loss.item():.4f} | KLD: {kl_loss_constrained.item():.4f} | KL Weight: {kl_weight:.2f}")
        
        logging.info("=== TRAINING COMPLETED ===\n")
        return history
    
    def inference(
            self, 
            x_past, 
            last_known_step, 
            horizon=11, 
            num_scenarios=5
        ):
        """
        Метод для генерации сценариев будущего на основе предыстории.
        
        Параметры:
        ----------
        x_past : torch.Tensor
            Тензор истории формы (batch_size, 5, feature_dim)
        last_known_step : torch.Tensor
            Тензор точки опоры формы (batch_size, feature_dim)
        horizon : int
            Сколько всего шагов генерировать суммарно (по умолчанию 11)
        num_scenarios : int
            Количество случайных альтернативных траекторий, которые нужно сгенерировать
            
        Возвращает:
        -----------
        scenarios : list of np.ndarray
            Список длины num_scenarios, где каждый элемент — массив формы (batch_size, horizon, feature_dim)
        """
        self.eval() # Обязательно переводим модель в режим оценки
        scenarios = []
        
        with torch.no_grad():
            # 1. Пропускаем через энкодер ОДИН раз, чтобы получить параметры распределения этой ситуации
            mu, log_var = self.encode(x_past)
            
            # 2. Генерируем несколько вариантов будущего за счет случайного сэмплирования
            for _ in range(num_scenarios):
                # Сэмплируем случайный латентный вектор Z
                z = self.reparameterize(mu, log_var)
                
                # Декодер разворачивает Z в дельты изменений
                delta = self.decode(z, horizon=horizon)
                
                # Прибавляем к базовой линии
                base_line = last_known_step.unsqueeze(1).repeat(1, horizon, 1)
                y_pred = base_line + delta
                
                # Сохраняем сценарий как NumPy массив для удобства визуализации
                scenarios.append(y_pred.cpu().numpy())
                
        return scenarios
