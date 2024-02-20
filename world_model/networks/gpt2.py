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

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

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
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
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

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 8                 # maximum sequence length
    vocab_size: int = 1104              # vocabulary size = visual_vocab_size + sum(action_vocab_sizes) (e.g. 1104 = 1024 + 50 + 30)
    nb_tokens_per_timestep: int  = 352  # nb_tokens_per_timestep = #visual-tokens + #action-tokens (e.g. 14*25 + 2)
    n_layer: int = 12                   # number of transformer blocks
    n_head: int = 16                    # number of attention heads
    n_embd: int = 256                   # embedding dimension for the world model    
    dropout: float = 0.15               # dropout rate
    bias: bool = True                   # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT2_Core(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.nb_tokens_per_timestep is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wie = nn.Embedding(config.vocab_size, config.n_embd),               # token embeddings
            wse = nn.Embedding(config.nb_tokens_per_timestep, config.n_embd),   # spatial position embeddings
            wte = nn.Embedding(config.block_size, config.n_embd),               # temporal position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

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

    def forward(self, tokens_sequence, inference=False):
        # Inputs:
        #   - token sequence of shape (b x t*(hw+act_sz))
        
        device = tokens_sequence.device
        b, nb_tokens_per_seq = tokens_sequence.size()
        assert nb_tokens_per_seq + 1 == self.config.block_size * self.config.nb_tokens_per_timestep
        
        # compute spatio-temporal position embeddings
        temporal_pos = torch.arange(0, self.config.block_size, dtype=torch.long, device=device)
        spatial_pos = torch.arange(0, self.config.nb_tokens_per_timestep, dtype=torch.long, device=device)
        temporal_pos_emb = self.transformer.wte(temporal_pos)                               # (t        x n_embd)
        spatial_pos_emb = self.transformer.wse(spatial_pos)                                 # (hw+act_sz x n_embd)
        spatio_temporal_emb = torch.einsum('td,nd->tnd', temporal_pos_emb, spatial_pos_emb) # (t x hw+act_sz x n_embd)
        spatio_temporal_emb = spatio_temporal_emb.view(-1, self.config.n_embd)[:-1,:]       # ((t*(hw+act_sz)-1) x n_embd)
        
        # compute image and action embeddings
        tok_emb = self.transformer.wie(tokens_sequence)                                     # token embeddings of shape (b x t x (hw+act_sz) x n_embd)
        emb_in = tok_emb + spatio_temporal_emb                                              # (b x t x (hw+act_sz) x n_embd)
        emb_in = emb_in.view(b, -1, self.config.n_embd)                                     # (b x t*(hw+act_sz) x n_embd)
        
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

class GPT2(nn.Module):
    
    def __init__(
        self, 
        embedding_dim,                  # embedding dimension for the world model
        num_heads,                      # number of attention heads
        num_layers,                     # number of transformer blocks
        vocabulary_size,                # vocabulary size = visual_vocab_size + sum(action_vocab_sizes)
        nb_timesteps,                   # maximum sequence length
        nb_tokens_per_timestep,         # nb_tokens_per_timestep = #visual-tokens + #action-tokens
        dropout_rate = 0.15,            # randomly dropout some pertange of the input tokens
        bias = True                     # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    ):
        super().__init__()
        config_args = dict(
            block_size=nb_timesteps,                 
            vocab_size=vocabulary_size,
            nb_tokens_per_timestep=nb_tokens_per_timestep, 
            n_layer=num_layers,                   
            n_head=num_heads,                    
            n_embd=embedding_dim,
            dropout=dropout_rate,
            bias=bias
        )
        configs = GPTConfig(**config_args)
        self.network = GPT2_Core(configs)
        
    def forward(self, tokens_sequence):
        return self.network.forward(tokens_sequence)
    
    def inference_forward(self, tokens_sequence):
        return self.network.forward(tokens_sequence, inference=True)