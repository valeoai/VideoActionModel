from typing import Tuple

import torch
from einops import rearrange, repeat


# adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
# meta took great care is correctly casting RoPE computation to avoid issue with FP16 & BF16
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_ = rearrange(x.float(), "... (d n) -> ... d n", n=2)
    x_ = torch.view_as_complex(x_)
    freqs_cis = freqs_cis[..., None, :, :]
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)

    return x_out.type_as(x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    freq_cis = precompute_freqs_cis(128, end=20, theta=100)
    print(freq_cis.shape)
    plt.imshow(torch.view_as_real(freq_cis)[..., 0])
    plt.show()

    ### check shift invariance
    ### dummy sequence of three frames [0, 0, 1, 1, 2, 2]
    ## with sliding window of size 2
    pos = torch.arange(2)
    pos = repeat(pos, "n -> b (s n)", b=3, s=2)  # 2 tokens per frames

    x = torch.randn(3, 2, 6, 128)  # b, n_head, seq_len, dim_head

    x_1 = apply_rotary_emb(x[:, :, :4, :], freq_cis[pos])
    x_2 = apply_rotary_emb(x[:, :, 2:, :], freq_cis[pos])

    print((x_1[0, :, 2] @ x_1[0, :, 3].T) == (x_2[0, :, 0] @ x_2[0, :, 1].T))

    print((x_1[0, :, 2] @ x_1[0, :, 2].T) == (x_2[0, :, 0] @ x_2[0, :, 0].T))
