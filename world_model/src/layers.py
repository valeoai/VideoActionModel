from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from world_model.src.kv_cache import KVCache
from world_model.src.pos_embedding import apply_rotary_emb


class TransformerLayer(nn.Module):
    def __init__(self, attn_dim: int, head_dim: int, ffn_expansion: int, layer_idx: int, ffn_dim: int = None) -> None:
        super().__init__()

        self.layer_idx = layer_idx

        if attn_dim % head_dim != 0:
            raise ValueError(f"attn_dim {attn_dim} is not divisible by head_size {head_dim}")

        if ffn_dim is None:
            ffn_dim = attn_dim

        self.head_dim = head_dim
        self.ffn_expansion = ffn_expansion

        self.norm = nn.RMSNorm(ffn_dim, elementwise_affine=False)
        self.qk_norm = nn.RMSNorm(head_dim, elementwise_affine=False)

        self.attn_qkv = nn.Linear(ffn_dim, 3 * attn_dim, bias=False)
        self.query = nn.Identity()  # just for coordcheck
        self.key = nn.Identity()  # just for coordcheck
        self.value = nn.Identity()  # just for coordcheck
        self.attn_out = nn.Linear(attn_dim, ffn_dim, bias=False)

        self.ffn_up = nn.Linear(ffn_dim, ffn_expansion * ffn_dim, bias=False)
        self.ffn_gate = nn.Linear(ffn_dim, ffn_expansion * ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_expansion * ffn_dim, ffn_dim, bias=False)

        self.out = nn.Identity()  # just for coordcheck

    def forward(self, x: Tensor, freqs_cis: Tensor, attn_mask: Optional[Tensor], kv_cache: Optional[KVCache] = None) -> Tensor:

        residual = self.norm(x)
        qkv = self.attn_qkv(residual)
        # scaled_dot_product_attention expect (b,...,h,t,d) as input shape for q, k and v
        q, k, v = rearrange(qkv, "b t (qkv h d) -> qkv b h t d", d=self.head_dim, qkv=3)

        ### muP: just for coord check (debug)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = self.qk_norm(q)
        k = self.qk_norm(k)

        # cache pre-rotary embedding to allow for sliding window inference
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        len_q = q.shape[-2]
        len_k = k.shape[-2]
        q_freqs_cis = freqs_cis[..., len_k - len_q : len_k, :]
        k_freqs_cis = freqs_cis[..., :len_k, :]

        q = apply_rotary_emb(q, q_freqs_cis)
        k = apply_rotary_emb(k, k_freqs_cis)

        ### muP: attention scaling 1/dim_head instead of 1/sqrt(dim_head)
        attn_scaling = 1 / v.size(-1)
        v = F.scaled_dot_product_attention(q, k, v, is_causal=False, attn_mask=attn_mask, scale=attn_scaling)

        residual = self.attn_out(rearrange(v, "b h t d -> b t (h d)"))
        x = x + residual

        residual = self.norm(x)
        residual = self.ffn_down(self.ffn_up(residual) * F.silu(self.ffn_gate(residual)))
        x = x + residual

        ### muP: just for coord check (debug)
        x = self.out(x)

        return x
