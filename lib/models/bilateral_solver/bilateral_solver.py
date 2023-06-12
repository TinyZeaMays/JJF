import torch
from torch import nn, Tensor
import numpy as np
from tqdm import tqdm


class BilateralSolver(nn.Module):
    def __init__(self,
                 reference: np.ndarray,
                 target: np.ndarray,
                 sigma_space: int = 8,
                 sigma_luma: int = 4,
                 lam: int = 128,
                 ) -> None:
        super().__init__()
        self.image_size = (reference.shape[1], reference.shape[0])

        position_x = torch.linspace(0, self.image_size[0], self.image_size[0])[None, :].repeat((self.image_size[1], 1))
        position_y = torch.linspace(0, self.image_size[1], self.image_size[1])[:, None].repeat((1, self.image_size[0]))
        position_xy = torch.stack([position_x, position_y], dim=-1)
        position_xy = position_xy.view((-1, 2))

        guide = torch.Tensor(reference).view((-1,))

        w_ij = torch.exp(
            - torch.sum((position_xy[:, None, :] - position_xy[None, :, :]) ** 2, dim=-1) / (2  * sigma_space ** 2)
            - (guide[:, None] - guide[None, :]) ** 2 / (2 * sigma_luma ** 2)
        )

        self.w_ij = nn.Parameter(w_ij, requires_grad=False)
        self.target = nn.Parameter(torch.Tensor(target).view((-1, 1)), requires_grad=False)
        self.output = nn.Parameter(torch.Tensor(np.rand_like(target)).view((-1, 1)), requires_grad=True)
        self.lam = lam

    def forward(self) -> Tensor:
        loss = self.lam / 2 * (self.output.T @ self.w_ij @ self.output) + torch.sum((self.output - self.target) ** 2)
        return loss


def bilateral_solver(reference: np.ndarray,
                     target: np.ndarray,
                     sigma_space: int = 8,
                     sigma_luma: int = 4,
                     lam: int = 128,
                     ) -> np.ndarray:

    solver = BilateralSolver(reference, target, sigma_space, sigma_luma, lam)
    optimizer = torch.optim.SGD(solver.parameters(), lr=1e-3, momentum=0.9)
    for _ in tqdm(range(10)):
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()

    output = torch.Tensor(solver.output).view((target.shape[0], target.shape[1])).cpu().numpy()

    return output


if __name__ == '__main__':
    import cv2
    refer = cv2.imread('reference.png', 0)
    tgt = cv2.imread('target.png', 0)

    out = bilateral_solver(refer, tgt)
    cv2.imwrite('result.png', out)







