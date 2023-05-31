from typing import Union, Tuple

import torch
from torch import Tensor, nn
from einops import rearrange

from lib.models.attention import MultiHeadSelfAttention


def patchify(images: Tensor, patch_size: Union[int, Tuple[int, int]]) -> Tensor:
    """convert input images into the patches

    :param images: tensor in the shape of (b, c, h, w)
    :param patch_size: width and height of the token
    :return: tensor of patches in the shape of (b, h // patch_h * w // patch_w, c * patch_h * patch_w)
    """
    b, c, h, w = images.shape
    patch_w, patch_h = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
    assert h % patch_h == 0 and w % patch_w == 0
    return rearrange(
        images, 'b c (dst_h patch_h) (dst_w patch_w) -> b (dst_h dst_w) (c patch_h patch_w)',
        patch_h=patch_h, patch_w=patch_w
    )


class AttentionBlock(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int, head_channels: int, mlp_ratio: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            MultiHeadSelfAttention(hidden_channels, num_heads, head_channels),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, int(hidden_channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_channels * mlp_ratio), hidden_channels)
        )

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attention(tokens)
        tokens = tokens + self.mlp(tokens)
        return tokens


class VisionTransformer(nn.Module):
    def __init__(self,
                 image_size: Union[int, Tuple[int, int]],
                 patch_size: Union[int, Tuple[int, int]],
                 out_channels: int,
                 hidden_channels: int,
                 num_blocks: int,
                 num_heads: int,
                 head_channels: int,
                 mlp_ratio: int = 4,
                 input_channels: int = 3,
                 ) -> None:
        super().__init__()
        image_w, image_h = (image_size, image_size) if isinstance(image_size, int) else image_size
        patch_w, patch_h = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.patch_size = (patch_w, patch_h)
        self.linear_mapper = nn.Sequential(
            nn.LayerNorm(patch_w * patch_h * input_channels),
            nn.Linear(patch_w * patch_h * input_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )

        num_tokens = image_w // patch_w * image_h // patch_h + 1
        self.class_token = nn.Parameter(torch.rand((1, 1, hidden_channels)))
        self.position_embedding = nn.Parameter(torch.rand(1, num_tokens, hidden_channels))

        self.blocks = nn.Sequential(
            *[AttentionBlock(hidden_channels, num_heads, head_channels, mlp_ratio) for _ in range(num_blocks)]
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, images: Tensor) -> Tensor:
        patches = patchify(images, self.patch_size)
        tokens = self.linear_mapper(patches)

        tokens = torch.cat([tokens, self.class_token.repeat(tokens.shape[0], 1, 1)], dim=1)
        tokens += self.position_embedding

        tokens = self.blocks(tokens)
        logits = self.mlp(tokens)
        return logits


if __name__ == "__main__":
    inp = torch.rand((16, 3, 256, 128))
    print(patchify(inp, patch_size=(16, 32)).shape)

    vit = VisionTransformer((256, 128), (16, 16), 10, 8, 3, 2, 16)
    print(vit(inp).shape)
