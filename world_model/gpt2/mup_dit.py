"""
A modified GPT for the adastra project on world model.
Original code: https://github.com/karpathy/nanoGPT/blob/master/model.py

Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

mup:
LayerNorm elementwise_affine=False  (see https://arxiv.org/abs/2404.05728)
MuReadout
MuSharedReadout
normal_ init
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from mup import MuReadout, MuSharedReadout, normal_
from torch import Tensor


class MLP(nn.Module):

    def __init__(self, dim_model: int, hidden_dim_mult: int = 4) -> None:
        super().__init__()
        self.c_fc = nn.Linear(dim_model, hidden_dim_mult * dim_model, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim_mult * dim_model, dim_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(
        self, dim_model: int, attn_dim: int, dim_heads: int, bias: bool, dropout: float, block_size: int, attn_scale: float
    ) -> None:
        super().__init__()
        assert dim_model % dim_heads == 0, "dim_model must be divisible by dim_heads"

        self.dim_heads = dim_heads
        self.dim_model = dim_model
        self.attn_dim = attn_dim

        ########### muP ###########
        self.attn_scale = attn_scale
        self.attn_score = nn.Identity()  # just for coordcheck
        self.query = nn.Identity()  # just for coordcheck
        self.key = nn.Identity()  # just for coordcheck
        self.value = nn.Identity()  # just for coordcheck

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(dim_model, 3 * self.attn_dim, bias=False)

        # output projection
        self.c_proj = nn.Linear(self.attn_dim, dim_model, bias=False)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x: Tensor, attn_mask: Tensor, start_pos: int = -1) -> Tensor:
        # calculate query, key, values for all heads in batch
        x = self.c_attn(x)

        # split into qkv and heads
        q, k, v = rearrange(x, "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads", n=3, dim_heads=self.dim_heads)

        # KV cache update
        if self.cache is not None:
            assert isinstance(start_pos, int), "start_pos must be an integer"
            # update the KV cache with current KV and get all the previous KVs
            k, v = self.cache.update(start_pos, k, v)

        ### muP: just for coord check (debug)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        ### muP: attention scaling 1/dim_heads instead of 1/sqrt(dim_heads)
        attn_scaling = self.attn_scale / v.size(-1)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=attn_scaling,  # muP: attention scaling
        )

        y = rearrange(y, "b nb_heads seq dim_head -> b seq (nb_heads dim_head)")  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y


class Block(nn.Module):

    def __init__(
        self,
        dim_model: int,
        attn_dim: int,
        dim_heads: int,
        block_size: int,
        attn_scale: float,
        mlp_dim_mult: int = 4,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(dim_model, attn_dim, dim_heads, block_size, attn_scale)
        self.ln_2 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.mlp = MLP(dim_model, mlp_dim_mult)

    def forward(self, x: Tensor, attn_mask: Tensor, start_pos: int = -1) -> Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask, start_pos=start_pos)
        x = x + self.mlp(self.ln_2(x))
        return x


class MupDiT(nn.Module):
    """
    GPT2 implementation following the original formulation to be able to load existing pre-trained weights

    Args:
        embedding_dim: embedding dimension for the world model
        nb_layers: number of transformer blocks
        dim_heads: dimension of attention heads
        vocabulary_size: total number of vector embeddings in GPT's codebook
        nb_timesteps: maximum number of timesteps found in one sequence
        nb_tokens_per_timestep: number of tokens ithi each timestep
        dropout_rate: dropout rate
        bias: True: bias in Linears. False: a bit better and faster
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 256,
        attention_dim: int = 256,
        nb_layers: int = 12,
        dim_heads: int = 2,
        mlp_dim_mult: int = 4,
        action_dim: int = 2,
        action_horizon: int = 6,
        context_length: int = 8,
        number_high_level_command: int = 3,
        max_period: float = 10000.0,
        init_std: float = 0.02,
        output_tied: bool = True,
        output_scale: float = 1.0,
        attn_scale: float = 1.0,
    ) -> None:

        super().__init__()

        self.embedding_dim = embedding_dim
        self.init_std = init_std
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.mlp_dim_mult = mlp_dim_mult
        self.nb_layers = nb_layers
        self.output_tied = output_tied

        self.transformer = nn.ModuleDict(
            {
                "ae": ActionEncoder(
                    action_dim,
                    embedding_dim,
                    context_length,
                    action_horizon,
                    number_high_level_command,
                    max_period=max_period,
                ),
                "h": nn.ModuleList(
                    [
                        Block(
                            embedding_dim,
                            attention_dim,
                            dim_heads,
                            self.block_size,
                            attn_scale,
                            mlp_dim_mult,
                        )
                        for _ in range(nb_layers)
                    ]
                ),
                "ln_f": nn.LayerNorm(embedding_dim, elementwise_affine=False),
            }
        )

        if output_tied:
            self.lm_head = MuSharedReadout(self.transformer.ae.linear_1.weight, bias=False, output_mult=output_scale)
        else:
            self.lm_head = MuReadout(embedding_dim, action_dim, bias=False, output_mult=output_scale)

        # init all weights
        ################## /!\ IMPORTANT READ ##################
        ### muP: swap constant std normal init with `normal_` from `mup.init`.
        ### Because `_init_weights` is called in `__init__`, before `infshape` is set,
        ### we need to manually call `self.apply(self._init_weights)` after calling
        ### `set_base_shape(model, base)`
        ###
        ### for proper muP init
        ### 1. instantiate model
        ### 2. call set_base_shape(model, base_shape.bsh)
        ### 3. reinit manually with model.apply(model._init_weights)
        self.apply(self._init_weights)

        # report number of parameters
        print("number of non-embedding parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.ae.command_embedding.weight.numel()
            n_params -= self.ae.action_positional_embedding.weight.numel()
            n_params -= self.ae.context_positional_embedding.weight.numel()
        if not self.output_tied:
            n_params -= self.transformer.ae.linear_1.weight.numel()
        return n_params

    def _init_c_proj_residual(self, module: nn.Module, is_mlp: bool) -> None:
        """
        Apply special scaled init to the residual projections (attn & mlp), per GPT-2 paper

            > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            >   -- GPT-2 :: https://openai.com/blog/better-language-models/

        This is combined with the muP initilization that is scaling init_std by 1/sqrt(width).

        Note: for the MLP c_proj, the scaling is 1/sqrt(width * mlp_hidden_mult)
        """
        # times 2 because in a block there are 2 residual paths, attn & mlp
        scaling = 2 * self.nb_layers * self.embedding_dim

        if is_mlp:
            scaling *= self.mlp_dim_mult

        depth_std = self.init_std * scaling**-0.5

        for p_name, p in module.named_parameters():
            if p_name.endswith("c_proj.weight"):
                p.data.normal_(mean=0.0, std=depth_std)

    def _init_weights(self, module: nn.Module) -> None:
        """
        This function is called using model.apply(model._init_weights)

        This will apply `_init_weights` recursively to every submodule.
        Children are seen first and then Parents.

        Example with the model=MLP(), if we print every module seen it gives:

            Linear(in_features=128, out_features=512, bias=False)
            ------------------------------
            GELU(approximate='none')
            ------------------------------
            Linear(in_features=512, out_features=128, bias=False)
            ------------------------------
            MLP(
            (c_fc): Linear(in_features=128, out_features=512, bias=False)
            (gelu): GELU(approximate='none')
            (c_proj): Linear(in_features=512, out_features=128, bias=False)
            )

        This means that specific initialization to MLP using isinstance(module, MLP)
        will override the initialization from isinstance(module, nn.Linear)

        We use this fact apply specific initialization rules in some modules.
        For example the init scaling of the MLP hidden nn.Linear layer is different
        than other nn.Linear layers.
        """

        # MuReadout zero init
        if isinstance(module, MuReadout) and not self.output_tied:
            # https://arxiv.org/abs/2404.05728 | 4.2.6 SP Unembedding Initialization
            # A simple alternative to either initialization is a zeroinitialized unembedding projection [15],
            # which we found to perform similarly to the µP initialization and to also facilitate transfer
            module.weight.data.zero_()

        elif isinstance(module, nn.Linear):

            if hasattr(module.weight, "infshape"):
                normal_(module.weight, mean=0.0, std=self.init_std)
            else:

                module.weight.data.normal_(mean=0.0, std=self.init_std * self.embedding_dim**-0.5)

            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

        elif isinstance(module, CausalSelfAttention):
            ### muP query zero init
            # yield equal attention weights over all past timesteps at initialization

            # if attention uses nn.Linear fanout is first dim
            # because nn.Linear applies y=xA.T+b
            # if using Conv1D as in hugginfaces transformers, fanout is last dim
            fanout, _ = module.c_attn.weight.shape
            assert fanout % 3 == 0  # assert matrix is used for query, key and value

            # if attention uses, nn.Linear change init in first dim
            # if using Conv1D, change init in last dim
            module.c_attn.weight.data[: fanout // 3, :] = 0

            self._init_c_proj_residual(module, is_mlp=False)

        elif isinstance(module, MLP):
            self._init_c_proj_residual(module, is_mlp=True)

    def forward(
        self,
        action_sequence: Tensor,
        diffusion_step: Tensor,
        high_level_command: Tensor,
        action_positions: Tensor,
        context_positions: Tensor,
        inference: bool = False,
        start_pos: int = -1,
    ) -> Tensor:
        x = self.transformer.ae(action_sequence, diffusion_step, high_level_command, action_positions, context_positions)

        seqlen = action_sequence.size(1)
        if inference and start_pos != -1:
            attn_mask = None
            if seqlen > 1:
                attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=action_sequence.device)
                attn_mask = torch.triu(attn_mask, diagonal=1)
                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                attn_mask = torch.hstack([torch.zeros((seqlen, start_pos), device=action_sequence.device), attn_mask]).type_as(
                    x
                )
        else:
            attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=action_sequence.device, dtype=torch.bool))

        # forward world embeddings to the transformer
        for block in self.transformer.h:
            x = block(x, attn_mask, start_pos=start_pos)
        emb_out = self.transformer.ln_f(x)

        if not inference:
            img_logits = self.lm_head(emb_out)
        else:
            img_logits = self.lm_head(emb_out[:, [-1], :])  # note: using list [-1] to preserve the time dim

        return img_logits


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(
        self,
        t: Tensor,
        max_period: float = 10000.0,
    ) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ActionEncoder(nn.Module):
    """Matching pi0 appendix"""

    def __init__(
        self,
        action_dim: int,
        width: int,
        context_length: int,
        action_horizon: int,
        number_high_level_command: int,
        max_period: float = 10000.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(action_dim, width, bias=bias)
        self.linear_2 = nn.Linear(2 * width, width, bias=bias)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(width, width, bias=bias)

        self.command_embedding = nn.Embedding(number_high_level_command, width)
        self.action_positional_embedding = nn.Embedding(action_horizon, width)
        self.context_positional_embedding = nn.Embedding(context_length, width)

        self.diffusion_step_embedding = SinusoidalPosEmb(width, max_period=max_period)

    def forward(
        self,
        action: Tensor,
        diffusion_step: Tensor,
        high_level_command: int,
        action_positions: Tensor,
        context_positions: Tensor,
    ) -> Tensor:
        # [Batch_Size, Seq_Len, Width]
        emb = self.linear_1(action)
        command_emb = self.command_embedding(high_level_command)
        action_emb = self.action_positional_embedding(action_positions)
        context_emb = self.context_positional_embedding(context_positions)
        emb = emb + command_emb + action_emb + context_emb
        # repeat time embedding for seq_len
        # [Batch_Size, Seq_Len, Width]
        diffusion_step_emb = self.diffusion_step_embedding(diffusion_step)
        diffusion_step_emb_full = diffusion_step_emb.unsqueeze(1).expand(-1, action.size(1), -1)
        emb = torch.cat([diffusion_step_emb_full, emb], dim=-1)
        emb = self.nonlinearity(self.linear_2(emb))
        emb = self.linear_3(emb)
        return emb
