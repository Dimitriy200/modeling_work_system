import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import logging
from typing import Optional, Dict, Tuple, Any


class ConditionalEncoder(nn.Module):
    """
    Encoder, который читает ТОЛЬКО контекст (первые k шагов последовательности).
    Сжимает контекст в параметры латентного распределения.
    """
    def __init__(
            self, 
            input_dim: int, # Сколько признаков (сенсоров) подается на каждом шаге
            hidden_dim: int, #  Размер "памяти" LSTM 
            latent_dim: int, # Размер латентного пространства
            n_layers: int = 2 # Количество слоев LSTM, стоящих друг на друге.
            ):
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
        Получаем Мат. ожидание и логарифм дисперсии входа

        Вход
        x_context
        (32, 5, 16)
        32 двигателя, 5 шагов, 16 сенсоров
        LSTM
        self.lstm(x)
        h_n: (2, 32, 64)
        LSTM накапливает информацию во времени
        
        Срез
        h_n[-1]
        h_last: (32, 64)
        Берем "мысли" последнего слоя LSTM
        
        Центр
        fc_mu(h_last)
        mu: (32, 16)
        Нейросеть думает: "Двигатели похожи на эти точки"
        
        Разброс
        fc_logvar(h_last)
        logvar: (32, 16)
        Нейросеть думает: "С такой-то уверенностью"
        """
        _, (h_n, _) = self.lstm(x_context)
        h_last = h_n[-1]  # Берем выход последнего слоя. Сжатое математическое представление того, как двигатели из батча вели себя в первые 5 циклов.
        
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class ConditionalDecoder(nn.Module):
    """
    Decoder, который генерирует полную последовательность на основе латентного вектора.
    """
    def __init__(
            self, 
            latent_dim: int, #  
            hidden_dim: int, # 
            output_dim: int, # 
            seq_len: int, # 
            n_layers: int = 2 # 
            ):
        
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # Всегда ожидаем на вход только латентный вектор. Латентный вектор z слишком мал (16), чтобы сразу кормить его в LSTM. Этот слой «расширяет» его до размера памяти LSTM
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, n_layers, 
            batch_first=True, 
            dropout=0.2 if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        """
        z: (batch_size, latent_dim)
        """
        # Проецируем и повторяем на всю длину последовательности. Добавляем измерение времени.
        z_expanded = self.fc_init(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Генерация последовательности через LSTM.
        # Выход: (32, 40, 64) — тензор, где на каждом временном шаге уже разные значения.
        out, _ = self.lstm(z_expanded)

        # Перевод в показания сенсоров
        return self.fc_out(out)


class Conditional_LSTM_VAE(nn.Module):
    """
    Conditional Variational Autoencoder для временных рядов.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 seq_len: int, context_len: int = 5, n_layers: int = 2):
        super().__init__()
        
        if context_len >= seq_len:
            raise ValueError(f"context_len ({context_len}) должен быть меньше seq_len ({seq_len})")
        
        self.context_len = context_len
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        
        # Исправлено: убран context_len из инициализации декодера
        self.encoder = ConditionalEncoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = ConditionalDecoder(latent_dim, hidden_dim, input_dim, seq_len, n_layers)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Трюк репараметризации для дифференцируемого сэмплирования."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_full: torch.Tensor):
        """
        x_full: (batch_size, seq_len, input_dim) — полная последовательность
        """
        # 1. Извлекаем контекст (первые k шагов)
        x_context = x_full[:, :self.context_len, :]
        
        # 2. Encoder сжимает контекст в латентное распределение
        mu, logvar = self.encoder(x_context)
        
        # 3. Сэмплируем латентный вектор
        z = self.reparameterize(mu, logvar)
        
        # 4. Decoder генерирует полную последовательность
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar

    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kld_loss
        return total_loss, recon_loss, kld_loss

    def generate(self, x_context: torch.Tensor, n_samples: int = 1, 
                temperature: float = 1.0) -> np.ndarray:
        """Генерация последовательностей по заданному контексту."""
        self.eval()
        device = next(self.parameters()).device
        x_context = x_context.to(device)
        
        generated_sequences = []
        
        with torch.no_grad():
            mu, logvar = self.encoder(x_context)
            
            for _ in range(n_samples):
                std = torch.exp(0.5 * logvar) * temperature
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                x_gen = self.decoder(z)
                generated_sequences.append(x_gen.cpu().numpy())
        
        return np.array(generated_sequences)

    def generate_from_noise(self, n_samples: int = 100, 
                           temperature: float = 1.0) -> np.ndarray:
        """Генерация последовательностей из чистого шума."""
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim).to(device)
            z = z * temperature
            x_gen = self.decoder(z)
            
        return x_gen.cpu().numpy()

    def fit(self, X_train: np.ndarray, X_val: np.ndarray, 
            epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, 
            warmup_epochs: int = 10, device: str = None):
        """Обучает Conditional VAE."""
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
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'train_loss': [], 'val_loss': [], 
            'train_recon': [], 'train_kld': []
        }
        
        logging.info(f"=== STARTING CONDITIONAL LSTM-VAE TRAINING on {device} ===")
        logging.info(f"Context length: {self.context_len}, Sequence length: {self.seq_len}")
        logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(1, epochs + 1):
            beta = min(1.0, epoch / warmup_epochs)
            
            self.train()
            epoch_train_loss, epoch_recon, epoch_kld = 0.0, 0.0, 0.0
            
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                
                x_recon, mu, logvar = self(x)
                loss, recon_loss, kld_loss = self.loss_function(x_recon, x, mu, logvar, beta=beta)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kld += kld_loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_recon = epoch_recon / len(train_loader)
            avg_kld = epoch_kld / len(train_loader)
            
            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_recon, mu, logvar = self(x)
                    loss, _, _ = self.loss_function(x_recon, x, mu, logvar, beta=beta)
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
            history['train_recon'].append(avg_recon)
            history['train_kld'].append(avg_kld)
            
            logging.info(
                f"Epoch {epoch:03d}/{epochs} | Beta: {beta:.2f} | "
                f"Train: {avg_train_loss:.4f} (Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}) | "
                f"Val: {avg_val_loss:.4f}"
            )
            
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}")
                self.load_state_dict(best_state)
                break
        
        logging.info("=== TRAINING FINISHED ===")
        return history
    

    