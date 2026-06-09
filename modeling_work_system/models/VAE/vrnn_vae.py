import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging



class TimeSeriesVRNN(nn.Module):
    def __init__(self, feature_dim, latent_dim, hidden_dim=64):
        super(TimeSeriesVRNN, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # 1. Слой извлечения признаков (Feature Extraction)
        self.phi_x = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        
        # 2. Энкодер (Prior & Posterior)
        # Априорное распределение ( Prior - зависит только от памяти LSTM )
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2) # Выдает mu и log_var
        )
        # Постериорное распределение ( Posterior - энкодер, видит и память, и текущий x )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2) # Выдает mu и log_var
        )
        
        # 3. Декодер (Восстановление/Предсказание следующего x)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # 4. Базовая рекуррентная сеть (Память системы)
        # На вход принимает скомбинированные признаки текущего x и текущего z
        self.rnn = nn.LSTMCell(hidden_dim + hidden_dim, hidden_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_sequence):
        """
        x_sequence: Тензор полного окна формы (batch_size, seq_len, feature_dim)
        Для обучения мы подаем все доступные шаги окна.
        """
        batch_size = x_sequence.size(0)
        seq_len = x_sequence.size(1)
        
        # Инициализируем скрытые состояния LSTMCell нулями
        h = torch.zeros(batch_size, self.hidden_dim, device=x_sequence.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=x_sequence.device)
        
        # Списки для накопления лоссов по каждому шагу
        total_mse = 0
        total_kld = 0
        
        # Идем по всей последовательности шаг за шагом
        for t in range(seq_len):
            x_t = x_sequence[:, t, :] # Текущий шаг (batch, feature_dim)
            
            # Извлекаем признаки из x_t
            phi_x_t = self.phi_x(x_t)
            
            # 1. Считаем PRIOR (что модель ожидает увидеть на основе памяти h)
            prior_params = self.prior(h)
            prior_mu, prior_log_var = torch.chunk(prior_params, 2, dim=-1)
            
            # 2. Считаем POSTERIOR (корректируем ожидания, глядя на реальный x_t)
            post_input = torch.cat([phi_x_t, h], dim=-1)
            post_params = self.posterior(post_input)
            post_mu, post_log_var = torch.chunk(post_params, 2, dim=-1)
            
            # 3. Сэмплируем латентный вектор Z_t для текущего шага
            z_t = self.reparameterize(post_mu, post_log_var)
            phi_z_t = self.phi_z(z_t)
            
            # 4. ДЕКОДЕР: Пытаемся восстановить x_t (или предсказать следующий)
            dec_input = torch.cat([phi_z_t, h], dim=-1)
            x_pred_t = self.decoder(dec_input)
            
            # 5. Считаем лоссы для текущего шага t
            mse = F.mse_loss(x_pred_t, x_t, reduction='mean')
            
            # KLD между Posterior и Prior распределениями на шаге t
            var_ratio = (post_log_var - prior_log_var).exp()
            t_mu = (prior_mu - post_mu).pow(2) / prior_log_var.exp()
            kld = 0.5 * (prior_log_var - post_log_var - 1 + var_ratio + t_mu).sum(dim=-1).mean()
            
            total_mse += mse
            total_kld += kld
            
            # 6. ОБНОВЛЯЕМ ПАМЯТЬ LSTM: подаем признаки x_t и z_t в ячейку
            rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
            h, c = self.rnn(rnn_input, (h, c))
            
        return total_mse / seq_len, total_kld / seq_len

    # ----------------------------------------------------
    # ИНФЕРЕНС С ДИНАМИЧЕСКИМ СЭМПЛИРОВАНИЕМ ШУМА
    # ----------------------------------------------------
    def inference(self, x_past, horizon=10, num_scenarios=5):
        """
        x_past: Известная история (batch, 5, feature_dim)
        """
        self.eval()
        batch_size = x_past.size(0)
        scenarios = []
        
        past_len = int(x_past.size(1) / 2)

        with torch.no_grad():
            for s in range(num_scenarios):
                h = torch.zeros(batch_size, self.hidden_dim, device=x_past.device)
                c = torch.zeros(batch_size, self.hidden_dim, device=x_past.device)
                
                generated_window = []
                
                # Этап 1: "Прогреваем" память LSTM реальной историей (шаги 1-5)
                for t in range(past_len):
                    x_t = x_past[:, t, :]
                    generated_window.append(x_t.unsqueeze(1))
                    
                    phi_x_t = self.phi_x(x_t)
                    post_input = torch.cat([phi_x_t, h], dim=-1)
                    post_params = self.posterior(post_input)
                    post_mu, post_log_var = torch.chunk(post_params, 2, dim=-1)
                    
                    z_t = self.reparameterize(post_mu, post_log_var)
                    phi_z_t = self.phi_z(z_t)
                    
                    rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
                    h, c = self.rnn(rnn_input, (h, c))
                
                # Этап 2: Чистая генерация будущего (шаги 6-10)
                # Модель генерирует x_t, опираясь ТОЛЬКО на PRIOR (свои предчувствия из памяти h)
                for t in range(past_len, horizon):
                    prior_params = self.prior(h)
                    prior_mu, prior_log_var = torch.chunk(prior_params, 2, dim=-1)
                    
                    # Сэмплируем шум будущего из априорного распределения
                    z_t = self.reparameterize(prior_mu, prior_log_var)
                    phi_z_t = self.phi_z(z_t)
                    
                    # Генерируем следующий шаг
                    dec_input = torch.cat([phi_z_t, h], dim=-1)
                    x_gen_t = self.decoder(dec_input)
                    
                    generated_window.append(x_gen_t.unsqueeze(1))
                    
                    # Фидбечим сгенерированный шаг обратно в память сети
                    phi_x_t = self.phi_x(x_gen_t)
                    rnn_input = torch.cat([phi_x_t, phi_z_t], dim=-1)
                    h, c = self.rnn(rnn_input, (h, c))
                    
                scenario_tensor = torch.cat(generated_window, dim=1)
                scenarios.append(scenario_tensor.cpu().numpy())
                
        return scenarios

    # Специальный метод fit под пошаговую структуру VRNN
    def fit(self, x_full_window, epochs=150, lr=0.001, tau=0.15, verbose_step=20):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'total_loss': [], 'mse_loss': [], 'kl_loss': [], 'kl_weight': []}
        
        logging.info(f"--- START VRNN TRAINING ({epochs} epochs) ---")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # VRNN сама считает MSE и KLD внутри forward, проходя по всему окну
            mse_loss, kl_loss = self.forward(x_full_window)
            
            # Защита Free Bits
            kl_loss_constrained = torch.clamp(kl_loss, min=tau)
            
            start_annealing = int(epochs * 0.3)
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
                logging.info(f"EPOCH {epoch:03d} | Loss: {total_loss.item():.4f} | MSE : {mse_loss.item():.4f} | KLD: {kl_loss_constrained.item():.4f}")
        return history
