from typing import Tuple, Optional
import os
import random

import torch
import torch.nn.functional as F

from diffuser.models.transition_head import TransitionHead
from diffuser.models.reward_head import RewardHead


# ------------------------ small utils ------------------------ #

def freeze_(module: torch.nn.Module) -> torch.nn.Module:
    """Put a module into eval mode and freeze its parameters."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


# ------------------------ path logic ------------------------ #

def auto_head_paths(
    args,
    env: Optional[str],
    horizon: int,
    n_steps: int,
    jump: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Build deterministic checkpoint paths for Transition/Reward heads.

    If args.transition_ckpt / args.reward_ckpt are provided and not 'auto',
    they take precedence.
    """
    root = getattr(args, "savepath", "logs/heads")
    env = env or getattr(args, "dataset", "env")
    jtag = f"_J{jump}" if jump is not None else ""
    tag = f"{env}_H{horizon}_T{n_steps}{jtag}"

    head_dir = os.path.join(root, "heads")
    os.makedirs(head_dir, exist_ok=True)

    trans_path = os.path.join(head_dir, f"transition_{tag}.pth")
    reward_path = os.path.join(head_dir, f"reward_{tag}.pth")

    user_t = getattr(args, "transition_ckpt", None)
    user_r = getattr(args, "reward_ckpt", None)
    if user_t not in (None, "", "auto"):
        trans_path = user_t
    if user_r not in (None, "", "auto"):
        reward_path = user_r

    return trans_path, reward_path


# ------------------------ build/load/save ------------------------ #

def build_heads(
    s_dim: int,
    a_dim: int,
    device: str = "cuda",
    learn_var: bool = False,
) -> Tuple[TransitionHead, RewardHead]:
    """Construct Transition/Reward heads on the given device."""
    T_hat = TransitionHead(s_dim, a_dim, learn_var=learn_var).to(device)
    R_hat = RewardHead(s_dim, a_dim).to(device)
    return T_hat, R_hat


def load_heads(
    T_hat: TransitionHead,
    R_hat: RewardHead,
    trans_path: str,
    reward_path: str,
    device: str = "cuda",
) -> Tuple[bool, bool]:
    """
    Load checkpoints if present.
    Returns (transition_loaded, reward_loaded).
    """
    trans_ok = os.path.isfile(trans_path)
    reward_ok = os.path.isfile(reward_path)
    if trans_ok:
        T_hat.load_state_dict(torch.load(trans_path, map_location=device))
    if reward_ok:
        R_hat.load_state_dict(torch.load(reward_path, map_location=device))
    return trans_ok, reward_ok


def save_heads(
    T_hat: TransitionHead,
    R_hat: RewardHead,
    trans_path: str,
    reward_path: str,
) -> None:
    """Save Transition/Reward head checkpoints."""
    os.makedirs(os.path.dirname(trans_path), exist_ok=True)
    torch.save(T_hat.state_dict(), trans_path)
    torch.save(R_hat.state_dict(), reward_path)


# ------------------------ SequenceDataset fallback ------------------------ #

def _sample_minibatch_from_sequence_dataset(dataset, batch_size: int):
    """
    Randomly sample one batch from SequenceDataset-like object.

    Expected dataset[i] -> (x, cond)
      x: [T, s_dim + a_dim] (or similar)
    We only need x here; cond is not used for head pretraining.
    """
    xs = []
    for _ in range(batch_size):
        x_i, _ = dataset[random.randrange(len(dataset))]
        x_i = torch.as_tensor(x_i)
        xs.append(x_i)
    x = torch.stack(xs, dim=0)  # [B, T, s+a]
    return x, None


# ------------------------ D4RL helpers (optional) ------------------------ #

def _maybe_load_q_dataset(env_name: Optional[str]):
    """
    Try to load D4RL qlearning_dataset(env) for supervised pretraining.
    Returns the dataset dict if successful, otherwise None.
    """
    if env_name is None:
        return None
    try:
        import gym  # noqa: F401
        import d4rl  # noqa: F401
        env = gym.make(env_name)
        ds = d4rl.qlearning_dataset(env)
        for k in ["observations", "actions", "rewards"]:
            assert k in ds, f"qlearning_dataset missing key: {k}"
        return ds
    except Exception as e:
        print(f"[heads] Skip D4RL supervision for '{env_name}': {e}")
        return None


def _sample_minibatch_from_qds(qds, batch_size: int, device: str = "cuda"):
    """
    Sample a supervised batch (s, a, s', r) from qlearning_dataset dict.
    Uses next_observations if present; otherwise uses shift by +1.
    """
    import numpy as np
    N = qds["observations"].shape[0]
    # Avoid the very last index if we need s_{t+1}
    idx = np.random.randint(0, max(1, N - 1), size=(batch_size,))

    s = torch.as_tensor(qds["observations"][idx], device=device).float()
    a = torch.as_tensor(qds["actions"][idx], device=device).float()
    if "next_observations" in qds:
        sp = torch.as_tensor(qds["next_observations"][idx], device=device).float()
    else:
        # Fallback: simple shift (not perfect across episode boundaries)
        nxt = (idx + 1).clip(max=N - 1)
        sp = torch.as_tensor(qds["observations"][nxt], device=device).float()
    r = torch.as_tensor(qds["rewards"][idx], device=device).float()
    return s, a, sp, r


# ------------------------ supervised pretraining ------------------------ #

def pretrain_heads(
    T_hat: TransitionHead,
    R_hat: RewardHead,
    dataset,
    device: str = "cuda",
    steps: int = 5000,
    lr: float = 3e-4,
    batch_size: int = 64,
    s_dim: int = None,
    a_dim: int = None,
    train_reward: bool = False,
    env_name: Optional[str] = None,
) -> None:
    """
    Supervised pretraining of T_hat and (optionally) R_hat.

    Priority:
      1) If D4RL qlearning_dataset is available for env_name, use it to train:
         - Transition: (s, a) -> s'
         - Reward:     (s, a) -> r_t  (only if train_reward=True)
      2) Otherwise, fall back to SequenceDataset for Transition only:
         - x: [B, T, s+a] -> (s_t, a_t, s_{t+1}) from rolling window

    Notes:
      * Reward pretraining is skipped if D4RL is unavailable.
      * This function updates the heads in-place.
    """
    assert s_dim is not None and a_dim is not None, \
        "pretrain_heads requires s_dim and a_dim."

    opt_T = torch.optim.Adam(T_hat.parameters(), lr=lr)
    opt_R = torch.optim.Adam(R_hat.parameters(), lr=lr)

    T_hat.train()
    R_hat.train()

    qds = _maybe_load_q_dataset(env_name) if env_name else None
    warned_reward = False

    def _norm_with_dataset(t: torch.Tensor, key: str):
        # normalize with the SAME normalizer as SequenceDataset
        x_np = t.detach().cpu().numpy()
        xn_np = dataset.normalizer(x_np, key)   # uses DatasetNormalizer
        return torch.as_tensor(xn_np, device=t.device, dtype=t.dtype)

    for it in range(steps):
        if qds is not None:
            # --- Use D4RL (s, a, s', r) supervision ---
            s, a, sp, r = _sample_minibatch_from_qds(
                    qds, batch_size, device=device)
            # normalize to match SequenceDataset space
            s_n  = _norm_with_dataset(s,  "observations")
            a_n  = _norm_with_dataset(a,  "actions")
            sp_n = _norm_with_dataset(sp, "observations")
            # Transition: (s, a) -> s'
            mu, _ = T_hat(s_n, a_n)
            loss_T = F.mse_loss(mu, sp_n)
            opt_T.zero_grad(set_to_none=True)
            loss_T.backward()
            opt_T.step()

            # Reward: (s, a) -> r_t (optional)
            if train_reward:
                r_pred = R_hat(s_n, a_n).view_as(r)
                loss_R = F.mse_loss(r_pred, r)
                opt_R.zero_grad(set_to_none=True)
                loss_R.backward()
                opt_R.step()
                if it % 200 == 0:
                    print(f"[heads] step={it} | L_T={loss_T.item():.4f} | L_R={loss_R.item():.4f}", flush=True)
            else:
                if it % 200 == 0:
                    print(f"[heads] step={it} | L_T={loss_T.item():.4f}", flush=True)

        else:
            # --- Fallback: SequenceDataset (Transition only) ---
            x, _ = _sample_minibatch_from_sequence_dataset(dataset, batch_size)  # [B, T, s+a]
            x = x.to(device).float()
            s = x[..., :s_dim]
            a = x[..., s_dim:s_dim + a_dim]
            s_next = torch.roll(s, shifts=-1, dims=1)  # aligned to T, last step ignored

            # Align to T-1 (drop last frame)
            s_in = s[:, :-1, :].reshape(-1, s_dim)
            a_in = a[:, :-1, :].reshape(-1, a_dim)
            s_tgt = s_next[:, :-1, :].reshape(-1, s_dim)

            mu, _ = T_hat(s_in, a_in)
            if getattr(T_hat, "learn_var", False):
                # Gaussian NLL
                loss_T = 0.5 * ((sp_n - mu).pow(2) * torch.exp(-logvar) \
                        + logvar).mean()
            else:
                loss_T = F.mse_loss(mu, sp_n)
            loss_T = F.mse_loss(mu, s_tgt)
            opt_T.zero_grad(set_to_none=True)
            loss_T.backward()
            opt_T.step()

            if train_reward and not warned_reward:
                print("[heads] train_reward=True but no D4RL dataset; skip RewardHead pretraining.", flush=True)
                warned_reward = True

            if it % 200 == 0:
                print(f"[heads] step={it} | L_T={loss_T.item():.4f}", flush=True)

    # Caller decides whether to freeze later (see ensure_heads).
    # We do not freeze here to give flexibility.


# ------------------------ one-stop helper ------------------------ #

def ensure_heads(
    args,
    dataset,
    device: str = "cuda",
    pretrain_if_missing: bool = True,
    steps: int = 5000,
    lr: float = 3e-4,
    batch_size: int = 64,
    learn_var: bool = False,
    train_reward: bool = False,
    s_dim: int = None,
    a_dim: int = None,
) -> Tuple[TransitionHead, RewardHead, str, str]:
    """
    Build T_hat/R_hat, load from checkpoints if present; otherwise pretrain (optional),
    save, and finally freeze (by default) before returning.

    Returns:
      T_hat, R_hat, trans_ckpt_path, reward_ckpt_path
    """
    # Resolve dims
    s_dim = s_dim if s_dim is not None else getattr(dataset, "observation_dim", None)
    a_dim = a_dim if a_dim is not None else getattr(dataset, "action_dim", None)
    assert s_dim is not None and a_dim is not None, "Unable to infer s_dim/a_dim."

    # Resolve paths
    jump = getattr(args, "jump", None)
    env_name = getattr(args, "dataset", None)
    trans_path, reward_path = auto_head_paths(
        args, env_name, args.horizon, args.n_diffusion_steps, jump
    )

    # Build and try loading
    T_hat, R_hat = build_heads(s_dim, a_dim, device=device, learn_var=learn_var)
    trans_ok, reward_ok = load_heads(T_hat, R_hat, trans_path, reward_path, device=device)

    # Pretrain if missing
    if (not trans_ok or not reward_ok) and pretrain_if_missing:
        print(f"[heads] pretrain ->\n  T: {trans_path}\n  R: {reward_path}", flush=True)
        pretrain_heads(
            T_hat, R_hat, dataset, device=device,
            steps=steps, lr=lr, batch_size=batch_size,
            s_dim=s_dim, a_dim=a_dim, train_reward=train_reward,
            env_name=env_name,
        )
        save_heads(T_hat, R_hat, trans_path, reward_path)

    # By default, freeze both heads for downstream training.
    # If you want to fine-tune R_hat during diffusion training,
    # comment out the freeze for R_hat and make sure your optimizer includes its params.
    freeze_(T_hat)
    freeze_(R_hat)

    return T_hat, R_hat, trans_path, reward_path

