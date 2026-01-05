# fallnet_arch.py
import torch.nn as nn, torch

class FallNet(nn.Module):
    def __init__(self,
                 input_size:  int  = 66,
                 hidden_size: int  = 128,
                 num_layers:  int  = 3,      # ← por defecto 3
                 dropout:     float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, bidirectional=False, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.head(h).squeeze(1)
