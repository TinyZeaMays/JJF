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

        p_ij = - torch.sum((position_xy[:, None, :] - position_xy[None, :, :]) ** 2, dim=-1, dtype=torch.float32) / (2 * sigma_space ** 2)
        del position_xy

        guide = torch.tensor(reference, dtype=torch.float32).view((-1,))
        g_ij = - (guide[:, None] - guide[None, :]) ** 2 / (2 * sigma_luma ** 2)
        del guide

        w_ij = torch.exp(
            p_ij + g_ij
        )

        self.w_ij = nn.Parameter(w_ij, requires_grad=False)
        self.target = nn.Parameter(torch.Tensor(target / 255.).view((-1, 1)), requires_grad=False)
        self.output = nn.Parameter(torch.Tensor(target / 255.).view((-1, 1)), requires_grad=True)
        self.lam = lam

    def forward(self) -> Tensor:
        loss = self.image_size[0] * self.image_size[1] * self.lam * \
               torch.mean(self.w_ij * (self.output.T - self.output) ** 2) \
               + torch.mean((self.output - self.target) ** 2)
        return loss


def bilateral_solver(reference: np.ndarray,
                     target: np.ndarray,
                     sigma_space: int = 8,
                     sigma_luma: int = 4,
                     lam: int = 128,
                     ) -> np.ndarray:

    solver = BilateralSolver(reference, target, sigma_space, sigma_luma, lam)
    solver.cuda()
    optimizer = torch.optim.SGD(solver.parameters(), lr=1e-3)
    for _ in tqdm(range(10000)):
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()
        print(loss)

    output = torch.Tensor(solver.output).view((target.shape[0], target.shape[1])).detach().cpu().numpy() * 255

    return output


if __name__ == '__main__':
    import cv2
    refer = cv2.imread('reference.png', 0)
    refer = cv2.resize(refer, (128, 128))
    tgt = cv2.imread('target.png', 0)
    tgt = cv2.resize(tgt, (128, 128))

    out = bilateral_solver(refer, tgt)
    cv2.imwrite('result.png', out)







