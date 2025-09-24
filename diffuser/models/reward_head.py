import torch, torch.nn as nn

class RewardHead(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)
