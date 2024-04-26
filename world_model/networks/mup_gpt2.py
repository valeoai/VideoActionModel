"""
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

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops import rearrange

from world_model.networks.gpt2 import MLP

from mup import MuReadout, MuSharedReadout, normal_


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, bias, dropout, block_size, attn_mult):
        super().__init__()
        assert n_embd % n_head == 0
        
        self.n_head = n_head
        self.n_embd = n_embd
        
        ########### muP ###########
        self.attn_mult = attn_mult
        self.attn_score = nn.Identity() # just for coordcheck
        self.query = nn.Identity() # just for coordcheck
        self.key = nn.Identity() # just for coordcheck
        self.value = nn.Identity() # just for coordcheck
        
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

        # calculate query, key, values for all heads in batch
        x = self.c_attn(x)
        
        # split into qkv and heads
        q, k, v = rearrange(x, "b seq (n n_head dim_head) -> n b n_head seq dim_head", n=3, n_head=self.n_head)
        
        ### muP: just for coord check (debug)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)
        
        ### muP: attention scaling
        attn_scaling = self.attn_mult / v.size(-1)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
                scale=attn_scaling ### muP: attention scaling
            )
        else:            
            ### muP: attention scaling
            att = (q @ k.transpose(-2, -1)) * attn_scaling
            
            ### muP no-op, but allows tracking for coord check
            att = self.attn_score(att)
            
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = rearrange(y, "b n_head seq dim_v -> b seq (n_head dim_v)") # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class Block(nn.Module):

    def __init__(self, n_embd, n_head, bias, dropout, block_size, attn_mult):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout, block_size, attn_mult)
        self.ln_2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MuGPT2(nn.Module):
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
        dropout_rate: float = 0.0,
        init_std: float = 0.02,
        bias: bool = True,   
        output_mult: float = 1.0,
        output_tied: bool = True      
    ):
        
        super().__init__()
        assert vocabulary_size is not None
        assert nb_tokens_per_timestep is not None
        assert nb_timesteps is not None
        
        self.init_std = init_std
        
        self.nb_timesteps = nb_timesteps
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        
        self.nb_layers = nb_layers
        self.block_size = nb_timesteps*nb_tokens_per_timestep
        self.embedding_dim = embedding_dim
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        self.output_tied=output_tied

        self.transformer = nn.ModuleDict(dict(
            wie = nn.Embedding(vocabulary_size, embedding_dim),         # token embeddings
            wse = nn.Embedding(nb_tokens_per_timestep, embedding_dim),  # spatial position embeddings
            wte = nn.Embedding(nb_timesteps, embedding_dim),            # temporal position embeddings
            drop = nn.Dropout(dropout_rate),
            h = nn.ModuleList([
                Block(embedding_dim, nb_heads, bias, dropout_rate, self.block_size,attn_mult)
                for _ in range(nb_layers)
            ]),
            ln_f = nn.LayerNorm(embedding_dim, bias=bias),
        ))
        
        if output_tied:
            self.lm_head = MuSharedReadout(self.transformer.wie.weight, bias=False, output_mult=output_mult)
        else:
            self.lm_head = MuReadout(embedding_dim, vocabulary_size, bias=False, output_mult=output_mult)
        
        # init all weights
        self.apply(self._init_weights)
        
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
        
        ################## /!\ IMPORTANT READ ##################
        ### muP: swap constant std normal init with normal_ from `mup.init`.
        ### Because `_init_weights` is called in `__init__`, before `infshape` is set,
        ### we need to manually call `self.apply(self._init_weights)` after calling
        ### `set_base_shape(model, base)` 
        ###
        ### for proper muP init
        ### 1. instantiate model and then 
        ### 2. set_base_shape(model, base)
        ### 3. reinit manually with model.apply(model._init_weights)
        
        
        # MuReadout zero init          
        if isinstance(module, MuReadout) and not self.output_tied:
            module.weight.data.zero_()
            
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            if hasattr(module.weight, 'infshape'):
                normal_(module.weight, mean=0.0, std=self.init_std)
            else:
                module.weight.data.normal_(mean=0.0, std=self.init_std)
                
            if module.bias is not None:
                module.bias.data.zero_()
                
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


        ### muP query zero init
        if isinstance(module, CausalSelfAttention):
            
            # if attention uses nn.Linear fanout is first dim
            # because nn.Linear applies y=xA.T+b
            # if using Conv1D as in hugginfaces transformers, fanout is last dim
            fanout, _ = module.c_attn.weight.shape
            assert fanout % 3 == 0 # assert matrix is used for query, key and value
            
            # if attention uses, nn.Linear change init in first dim
            # if using Conv1D, change init in last dim
            module.c_attn.weight.data[:fanout//3, :] = 0
            
            
        # Apply special scaled init to the residual projections (attn & mlp), per GPT-2 paper
        #
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        depth_std = self.init_std / math.sqrt(2 * self.nb_layers)
        
        for p_name, p in module.named_parameters():
            if p_name.endswith('c_proj.weight'):
                ### muP Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                if hasattr(p, 'infshape'):
                    normal_(p, mean=0.0, std=depth_std)
                else:
                    p.data.normal_(mean=0.0, std=depth_std)
                    

    def forward(self, token_sequence, spatial_positions, temporal_positions, inference=False):
        """
        Args:
            token_sequence: A tensor of interleaved visual and action tokens.
            spatial_positions: A tensor indicating the spatial position of each token in the sequence. 
                example: [0,1,2,3,0,1,2,3]
            temporal_positions: A tensor indicating the temporal position of each token in the sequence.
                example: [0,0,0,0,1,1,1,1]
        """
        
        assert spatial_positions.max() <  self.nb_tokens_per_timestep
        assert temporal_positions.max() <  self.nb_timesteps
        
        # compute spatio-temporal position embeddings
        spatial_pos_emb = self.transformer.wse(spatial_positions)
        temporal_pos_emb = self.transformer.wte(temporal_positions)
        
        
        tok_emb = self.transformer.wie(token_sequence)
        
        emb_in = tok_emb + temporal_pos_emb + spatial_pos_emb         
        
        # forward world embeddings to the transformer
        x = self.transformer.drop(emb_in)
        for block in self.transformer.h:
            x = block(x)
        emb_out = self.transformer.ln_f(x)
        
        if not inference:
            img_logits = self.lm_head(emb_out)
        else:
            img_logits = self.lm_head(emb_out[:, [-1], :]) # note: using list [-1] to preserve the time dim
            
        return img_logits
