from typing import List, Union, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from tqdm import tqdm

from lib.utils import right_broadcast_like


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 batch_size: int,
                 image_size: Union[Tuple[int, int], int],
                 channels: int,
                 betas: Tensor,
                 loss_type: str = 'l2',
                 ) -> None:
        super().__init__()
        self.shape = [batch_size, channels, image_size, image_size] if isinstance(image_size, int) \
            else [batch_size, channels, image_size[1], image_size[0]]
        self.loss_type = loss_type
        self.num_timesteps = len(betas)

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

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_variance = nn.Parameter(posterior_variance, requires_grad=False)

        self.model = lambda x, t: x

    def q_sample(self, x_start: Tensor, timesteps: Union[Tensor, List[int]], noise: Tensor = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = right_broadcast_like(self.sqrt_alphas_cumprod[timesteps], x_start)

        sqrt_one_minus_alphas_cumprod_t = right_broadcast_like(self.sqrt_one_minus_alphas_cumprod[timesteps], x_start)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x: Tensor, timesteps: Union[Tensor, List[int]], add_variance: bool) -> Tensor:
        betas_t = right_broadcast_like(self.betas[timesteps], x)
        sqrt_one_minus_alphas_cumprod_t = right_broadcast_like(self.sqrt_one_minus_alphas_cumprod[timesteps], x)
        sqrt_recip_alphas_t = right_broadcast_like(self.sqrt_recip_alphas[timesteps], x)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, timesteps) / sqrt_one_minus_alphas_cumprod_t
        )
        if not add_variance:
            return model_mean
        else:
            posterior_variance_t = right_broadcast_like(self.posterior_variance[timesteps], x)
            noise = torch.randn_like(x)
            return model_mean + noise * torch.sqrt(posterior_variance_t)

    def p_sample_loop(self) -> List[np.ndarray]:
        device = self.betas.device
        batch_size = self.shape[0]
        image = torch.randn(self.shape, device=self.betas.device)
        images = []

        for idx in tqdm(reversed(range(0, self.num_timesteps))):
            image = self.p_sample(
                image, torch.full((batch_size,), idx, device=device, dtype=torch.long), add_variance=idx != 0
            )
            images.append(image.cpu().numpy())
        return images

    def p_loss(self, x_start: Tensor, timesteps: Union[Tensor, List[int]], noise: Tensor = None) -> Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, timesteps, noise)
        predicted_noise = self.model(x_noisy, timesteps)

        if self.loss_type == 'l1':
            loss = nn.functional.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = nn.functional.mse_loss(noise, predicted_noise)
        elif self.loss_type == "huber":
            loss = nn.functional.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError

        return loss


if __name__ == '__main__':
    from lib.models.diffusion.beta_schedule import linear_beta_schedule

    linear_betas = linear_beta_schedule(100)
    ddpm = GaussianDiffusion(4, (256, 256), 3, linear_betas)
    print(ddpm.p_loss(torch.rand(4, 3, 256, 128), [1, 2, 3, 4]))
    print(ddpm.p_sample_loop()[-1].shape)
