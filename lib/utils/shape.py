from torch import Tensor


def right_broadcast_like(src: Tensor, dst: Tensor) -> Tensor:
    num = len(dst.shape) - len(src.shape)
    out = src
    if num > 0:
        for _ in range(num):
            out = out.unsqueeze(-1)
    return out
