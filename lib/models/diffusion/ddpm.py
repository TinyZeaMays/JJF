from typing import List, Union

import torch
from torch import nn, Tensor

from lib.utils import right_broadcast_like


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 betas: Tensor,
                 ) -> None:
        super().__init__()

        self.betas = nn.Parameter(betas, requires_grad=False)

        # define alphas
        alphas = 1. - betas
        self.alphas = nn.Parameter(alphas, requires_grad=False)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod = nn.Parameter(alphas_cumprod, requires_grad=False)

        alphas_cumprod_prev = nn.functional.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.alphas_cumprod_prev = nn.Parameter(alphas_cumprod_prev, requires_grad=False)

        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_recip_alphas = nn.Parameter(sqrt_recip_alphas, requires_grad=False)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_alphas_cumprod = nn.Parameter(sqrt_alphas_cumprod, requires_grad=False)

        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = nn.Parameter(sqrt_one_minus_alphas_cumprod, requires_grad=False)

    def q_sample(self, x_start: Tensor, timesteps: Union[Tensor, List[int]], noise: Tensor = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = right_broadcast_like(self.sqrt_alphas_cumprod[timesteps], x_start)

        sqrt_one_minus_alphas_cumprod_t = right_broadcast_like(self.sqrt_one_minus_alphas_cumprod[timesteps], x_start)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


if __name__ == '__main__':
    from lib.models.diffusion.beta_schedule import linear_beta_schedule
    linear_betas = linear_beta_schedule(100)
    ddpm = GaussianDiffusion(linear_betas)
    print(ddpm.q_sample(torch.rand(4, 3, 256, 128), [1, 2, 3, 4]).shape)
