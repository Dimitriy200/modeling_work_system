import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import logging

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, n_layers: int = 2):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]
        return self.fc_mu(h_last), self.fc_logvar(h_last)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, seq_len: int, n_layers: int = 2):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers, batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z_expanded = self.fc_init(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(z_expanded)
        return self.fc_out(out)


class LSTM_VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, seq_len: int, n_layers: int = 2):
        super(LSTM_VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_layers)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_len, n_layers)
        self.seq_len = seq_len

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss, recon_loss, kld_loss

    # ==========================================================
    # НОВЫЙ МЕТОД FIT
    # ==========================================================
    def fit(self, X_train: np.ndarray, X_val: np.ndarray, 
            epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, 
            warmup_epochs: int = 10, device: str = None):
        """
        Обучает модель на предоставленных данных.
        
        Args:
            X_train: numpy array формы (N, seq_len, features)
            X_val: numpy array формы (N, seq_len, features)
            epochs: количество эпох
            batch_size: размер батча
            lr: скорость обучения
            warmup_epochs: эпохи для KL-Annealing (рост beta от 0 до 1)
            device: 'cuda' или 'cpu'
            
        Returns:
            history: словарь со списком потерь для построения графиков
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # 1. Создание DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32)), 
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32)), 
            batch_size=batch_size, shuffle=False
        )
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'train_kld': []}
        
        logging.info(f"=== STARTING LSTM-VAE TRAINING on {device} ===")
        logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # 2. Цикл обучения
        for epoch in range(1, epochs + 1):
            beta = min(1.0, epoch / warmup_epochs)
            
            # --- TRAIN ---
            self.train()
            epoch_train_loss, epoch_recon, epoch_kld = 0.0, 0.0, 0.0
            
            for batch in train_loader:
                x = batch[0].to(device)
                optimizer.zero_grad()
                
                x_recon, mu, logvar = self(x)
                loss, recon_loss, kld_loss = self.loss_function(x_recon, x, mu, logvar, beta=beta)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # Защита LSTM
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kld += kld_loss.item()
                
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_recon = epoch_recon / len(train_loader)
            avg_kld = epoch_kld / len(train_loader)
            
            # --- VAL ---
            self.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    x_recon, mu, logvar = self(x)
                    loss, _, _ = self.loss_function(x_recon, x, mu, logvar, beta=beta)
                    epoch_val_loss += loss.item()
                    
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            # Сохранение в историю
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_recon'].append(avg_recon)
            history['train_kld'].append(avg_kld)
            
            # Логирование
            logging.info(
                f"Epoch {epoch:03d}/{epochs} | Beta: {beta:.2f} | "
                f"Train: {avg_train_loss:.4f} (Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}) | "
                f"Val: {avg_val_loss:.4f}"
            )
                
        logging.info("=== TRAINING FINISHED ===")
        return history