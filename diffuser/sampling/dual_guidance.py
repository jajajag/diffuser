import torch

def _split_tau(tau0, s_dim, a_dim):
    s = tau0[..., :s_dim]
    a = tau0[..., s_dim:s_dim + a_dim]
    s_next = torch.roll(s, shifts=-1, dims=1)
    return s, a, s_next

def n_step_dual_guided_p_sample(
    model, x, cond, t, cond_key, *,
    T_hat, R_hat, s_dim, a_dim,
    alpha=0.1, lambda_tr_guide=1.0,
    n_guide_steps=1, t_stopgrad=0, scale_grad_by_std=True, **kwargs
):
    x = x.detach().requires_grad_(True)

    model_mean, _, model_logvar, x0_hat = model.p_mean_variance(
        x=x, cond=cond, t=t, return_x_recon=True
    )
    std = torch.exp(0.5 * model_logvar)
    scale = x0_hat.std(dim=(0,1), keepdim=True).clamp_min(1e-3) if scale_grad_by_std else 1.0

    s, a, s_next = _split_tau(x0_hat, s_dim, a_dim)
    B, T, _ = s.shape
    r = R_hat(s.reshape(B*T, s_dim), a.reshape(B*T, a_dim)).view(B, T).sum(dim=1)

    st, at, stp1 = s[:, :-1, :], a[:, :-1, :], s_next[:, :-1, :]
    out = T_hat(st.reshape(-1, s_dim), at.reshape(-1, a_dim))
    if isinstance(out, (tuple, list)) and len(out) == 2:
        mu_T, logvar_T = out
        mu_T, logvar_T = mu_T.view_as(stp1), logvar_T.view_as(stp1)
        diff = stp1 - mu_T
        logp_T = -0.5 * (diff.pow(2) * torch.exp(-logvar_T) + logvar_T).sum(dim=[1,2])
    else:
        mu_T = out.view_as(stp1)
        logp_T = -0.5 * (stp1 - mu_T).pow(2).sum(dim=[1,2])

    J_total = (r + lambda_tr_guide * logp_T).sum()
    g = torch.autograd.grad(J_total, x, retain_graph=False, create_graph=False)[0] / scale
    for _ in range(max(0, n_guide_steps - 1)):
        g = g + g

    mu_guided = model_mean + alpha * (std ** 2) * g
    noise = torch.randn_like(x)
    if isinstance(t, torch.Tensor):
        noise[t == 0] = 0
    elif t == 0:
        noise.zero_()
    x_next = mu_guided + std * noise
    with torch.no_grad():
        values = r.detach()
    return x_next.detach(), values
