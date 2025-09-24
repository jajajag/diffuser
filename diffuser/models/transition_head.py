import torch, torch.nn as nn

class TransitionHead(nn.Module):
    def __init__(self, s_dim, a_dim, hidden=512, learn_var=False):
        super().__init__()
        out = s_dim if not learn_var else 2 * s_dim
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out),
        )
        self.s_dim, self.learn_var = s_dim, learn_var

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        out = self.net(x)
        if not self.learn_var:
            mu = out
            logvar = torch.zeros_like(mu)
        else:
            mu, logvar = out.split(self.s_dim, dim=-1)
            logvar = logvar.clamp(-8, 8)
        return mu, logvar
