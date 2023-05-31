import torch
from torch import Tensor, nn
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_channels: int, num_heads: int, head_channels: int) -> None:
        super().__init__()
        self.head_channels = head_channels
        self.inner_channels = num_heads * head_channels

        self.qkv_mapper = nn.Linear(input_channels, self.inner_channels * 3)
        self.out_mapper = nn.Identity() if self.inner_channels == input_channels \
            else nn.Linear(self.inner_channels, input_channels)

    def forward(self, tokens: Tensor) -> Tensor:
        query, key, value = [
            rearrange(
                x, 'b num_tokens (num_heads head_channels) -> b num_heads num_tokens head_channels',
                head_channels=self.head_channels
            )
            for x in torch.split(self.qkv_mapper(tokens), self.inner_channels, dim=-1)]
        key = rearrange(key, 'b num_heads num_tokens head_channels -> b num_heads head_channels num_tokens')
        attention = torch.softmax(torch.matmul(query, key) / (self.head_channels ** 0.5), dim=-1)
        value = torch.matmul(attention, value)
        value = rearrange(value, 'b num_heads num_tokens head_channels -> b num_tokens (num_heads head_channels)')
        return self.out_mapper(value)


if __name__ == '__main__':
    inp = torch.rand((16, 129, 8))
    msa = MultiHeadSelfAttention(8, 2, 16)
    print(msa(inp).shape)
