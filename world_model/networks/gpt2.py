"""
Copyright: valeo.ai
Author: Tuan-Hung Vu
A modified GPT for the adastra project on world model.
Original code: https://github.com/karpathy/nanoGPT/blob/master/model.py

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# Building blocks for GPT-2: LayerNorm, SelfAttention, MLP, TransformerBlock
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    """
    GPT2 implementation following the original formulation to be able to load existing pre-trained weights    
    
    Args:
        embedding_dim: embedding dimension for the world model 
        nb_layers: number of transformer blocks
        nb_heads: number of attention heads
        vocabulary_size: total number of vector embeddings in GPT's codebook
        nb_timesteps: maximum number of timesteps found in one sequence
        nb_tokens_per_timestep: number of tokens ithi each timestep
        dropout_rate: dropout rate
        bias: True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    """
    def __init__(self, 
        embedding_dim: int = 256,
        nb_layers: int = 12,
        nb_heads: int = 16,
        vocabulary_size: int = 1104,
        nb_timesteps: int = 8,
        nb_tokens_per_timestep: int  = 352,
        dropout_rate: float = 0.15,
        bias: bool = True,             
    ):
        
        super().__init__()
        assert vocabulary_size is not None
        assert nb_tokens_per_timestep is not None
        assert nb_timesteps is not None
        
        self.nb_timesteps = nb_timesteps
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        
        self.block_size = nb_timesteps*nb_tokens_per_timestep
        self.embedding_dim = embedding_dim
        self.nb_tokens_per_timestep = nb_tokens_per_timestep

        self.transformer = nn.ModuleDict(dict(
            wie = nn.Embedding(vocabulary_size, embedding_dim),               # token embeddings
            wse = nn.Embedding(nb_tokens_per_timestep, embedding_dim),   # spatial position embeddings
            wte = nn.Embedding(nb_timesteps, embedding_dim),               # temporal position embeddings
            drop = nn.Dropout(dropout_rate),
            h = nn.ModuleList([
                Block(embedding_dim, nb_heads, bias, dropout_rate, self.block_size)
                for _ in range(nb_layers)
            ]),
            ln_f = LayerNorm(embedding_dim, bias=bias),
        ))
        self.lm_head = nn.Linear(embedding_dim, vocabulary_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # https://paperswithcode.com/method/weight-tying
        self.transformer.wie.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * nb_layers))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wse.weight.numel()
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_sequence, temporal_positions, spatial_positions, inference=False):
        """
        Args:
            token_sequence: A tensor of interleaved visual and action tokens.
            spatial_positions: A tensor indicating the spatial position of each token in the sequence. 
                example: [0,1,2,3,0,1,2,3]
            temporal_positions: A tensor indicating the temporal position of each token in the sequence.
                example: [0,0,0,0,1,1,1,1]
        """
        
        b, nb_tokens_per_seq = token_sequence.size()
        
        # compute spatio-temporal position embeddings
        temporal_pos_emb = self.transformer.wte(temporal_positions)
        spatial_pos_emb = self.transformer.wse(spatial_positions)
        
        # compute image and action embeddings
        tok_emb = self.transformer.wie(token_sequence)                                      # token embeddings of shape (b x t x (hw+act_sz) x n_embd)
        emb_in = tok_emb + temporal_pos_emb + spatial_pos_emb                               # (b x t x (hw+act_sz) x n_embd)
        emb_in = emb_in.view(b, -1, self.embedding_dim)                                     # (b x t*(hw+act_sz) x n_embd)
        
        # forward world embeddings to the transformer
        x = self.transformer.drop(emb_in)
        for block in self.transformer.h:
            x = block(x)
        emb_out = self.transformer.ln_f(x)                                                  # (b x t*(hw+act_sz) x n_embd)
        
        if not inference:
            img_logits = self.lm_head(emb_out)
        else:
            img_logits = self.lm_head(emb_out[:, [-1], :]) # note: using list [-1] to preserve the time dim
            
        return img_logits