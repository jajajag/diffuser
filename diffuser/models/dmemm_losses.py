import torch
import torch.nn.functional as F

def mse(a, b):  # [B,T,D] -> [B,T]
    return F.mse_loss(a, b, reduction='none').mean(dim=-1)

def reconstruct_x0(tau_k, eps_pred, sqrt_bar_alpha_k, sqrt_one_minus_bar_alpha_k):
    # x0_hat = (x_k - sqrt(1 - bar_alpha_k)*eps_pred) / sqrt(bar_alpha_k)
    return (tau_k - sqrt_one_minus_bar_alpha_k * eps_pred) / sqrt_bar_alpha_k

def split_tau(tau0, s_dim, a_dim):
    s = tau0[..., :s_dim]
    a = tau0[..., s_dim:s_dim+a_dim]
    s_next = torch.roll(s, shifts=-1, dims=1)  # 最后一拍可忽略
    return s, a, s_next

def transition_loss(tau0_hat, T_hat, s_dim, a_dim):
    s, a, s_next = split_tau(tau0_hat, s_dim, a_dim)
    mu, logvar = T_hat(s[:, :-1, :], a[:, :-1, :])
    diff = (s_next[:, :-1, :] - mu)
    l2 = (diff.pow(2) * torch.exp(-logvar)).sum(dim=-1) + logvar.sum(dim=-1)
    return l2.mean()

def reward_sum_hat(tau0_hat, R_hat, s_dim, a_dim):
    s, a, _ = split_tau(tau0_hat, s_dim, a_dim)
    return R_hat(s, a).sum(dim=1)  # [B]

def reward_mod_loss(tau0_hat, R_hat, s_dim, a_dim):
    return -reward_sum_hat(tau0_hat, R_hat, s_dim, a_dim).mean()

def reward_aware_diff_loss(eps, eps_pred, ret_norm):
    # ret_norm ∈ [0,1], shape [B]
    per = (eps - eps_pred).pow(2).flatten(1).mean(1)  # [B]
    return (ret_norm * per).mean()
