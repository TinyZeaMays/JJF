import torch
from torch import Tensor


def linear_beta_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps)
