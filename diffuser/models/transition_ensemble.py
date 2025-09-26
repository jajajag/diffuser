import torch, torch.nn as nn
from .transition_head import TransitionHead

class TransitionEnsemble(nn.Module):
    def __init__(self, s_dim, a_dim, M=5, hidden=256, learn_var=True):
        super().__init__()
        self.members = nn.ModuleList([
            TransitionHead(s_dim, a_dim, hidden=hidden, learn_var=learn_var)
            for _ in range(M)
        ])
        self.M = M
        self.s_dim = s_dim

    def forward_members(self, s, a):
        # returns lists of mus, logvars (each [B,*,s_dim])
        mus, logvars = [], []
        for m in self.members:
            mu, logvar = m(s, a)
            logvars.append(logvar)
            mus.append(mu)
        return mus, logvars

    def forward(self, s, a):
        """Moment-matched single Gaussian for convenience at inference:
           \bar{mu} = mean(mu_m),  \bar{var} = mean(var_m + mu_m^2) - \bar{mu}^2
        """
        mus, logvars = self.forward_members(s, a)
        mu = torch.stack(mus, dim=0)                  # [M, B,*, s_dim]
        var = torch.stack([lv.exp() for lv in logvars], dim=0)  # [M,B,*,s_dim]
        mu_bar = mu.mean(0)
        second = (var + mu**2).mean(0)
        var_bar = (second - mu_bar**2).clamp_min(1e-8)
        return mu_bar, var_bar.log()
