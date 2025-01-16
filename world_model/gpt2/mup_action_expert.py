"""
mup:
LayerNorm elementwise_affine=False  (see https://arxiv.org/abs/2404.05728)
MuReadout
MuSharedReadout
normal_ init
"""

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from mup import MuReadout, normal_
from torch import Tensor


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding the diffusion step.
    """

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
    """
    Embedding for actions.

    - Encodes the noisy actions
    - Embeds the high level command
    - Add positional embedding for the action horizon

    - Combines it with the diffusion step temporal embedding

    Parameters
    ----------
    action_dim: int, dimension of the action space
    action_hidden_dim: int, dimension of the action embedding and width of the denoiser
    action_horizon: int, number of timesteps in the action sequence,
                         not the same as the context length of the video prediction model.
    number_high_level_command: int, number of high level commands in the nuScenes / nuPlan dataset
    max_period: float, maximum period for the sinusoidal positional embedding
    bias: bool, whether to use bias in the linear layers
    """

    def __init__(
        self,
        action_dim: int,
        action_hidden_dim: int,
        action_horizon: int,
        number_high_level_command: int,
        max_period: float = 10000.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_hidden_dim = action_hidden_dim

        self.linear_1 = nn.Linear(action_dim, action_hidden_dim, bias=bias)
        self.linear_2 = nn.Linear(2 * action_hidden_dim, action_hidden_dim, bias=bias)
        self.nonlinearity = nn.SiLU()  # swish
        self.linear_3 = nn.Linear(action_hidden_dim, action_hidden_dim, bias=bias)

        self.command_embedding = nn.Embedding(number_high_level_command, action_hidden_dim)
        self.action_positional_embedding = nn.Embedding(action_horizon, action_hidden_dim)

        self.max_period = max_period
        self.diffusion_step_embedding = SinusoidalPosEmb(action_hidden_dim)

    def forward(
        self,
        actions: Tensor,
        high_level_command: Tensor,
        diffusion_step: Tensor,
    ) -> Tensor:
        # actions: [Batch_Size, timesteps, Horizon_Steps, Action_Dim]
        # high_level_command: [Batch_Size, context_length]
        # diffusion_step: [Batch_Size, context_length]
        bs, context_length, horizon, _ = actions.size()
        action_emb = self.linear_1(actions)  # [Batch_Size, context_length, Horizon_Steps, Action_Hidden_Dim]
        # embedd high level command
        command_emb = self.command_embedding(high_level_command)  # [Batch_Size, context_length, Action_Hidden_Dim]
        command_emb = repeat(command_emb, "b t d -> b t h d", h=horizon)
        # Pos embedding for actions
        action_pos_emb = self.action_positional_embedding(
            torch.arange(horizon, device=actions.device)
        )  # [Horizon_Steps, Action_Hidden_Dim]
        action_pos_emb = repeat(action_pos_emb, "h d -> b t h d", b=bs, t=context_length)
        # Final timstep action embedding
        action_emb = action_emb + command_emb + action_emb

        # Pos embedding for diffusion step
        diffusion_step = rearrange(diffusion_step, "b t -> (b t)")
        diffusion_step_emb = self.diffusion_step_embedding(
            diffusion_step, self.max_period
        )  # [Batch_Size, context_length, Action_Hidden_Dim]
        diffusion_step_emb = rearrange(diffusion_step_emb, "(b t) d -> b t d", b=bs)
        diffusion_step_emb = repeat(diffusion_step_emb, "b t d -> b t h d", h=horizon)

        action_emb = torch.cat([diffusion_step_emb, action_emb], dim=-1)
        action_emb = self.nonlinearity(self.linear_2(action_emb))
        action_emb = self.linear_3(action_emb)
        return action_emb  # [Batch_Size, timesteps, Horizon_Steps, Action_Hidden_Dim]


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


class SelfAttention(nn.Module):

    def __init__(self, dim_model: int, attn_dim: int, dim_heads: int, attn_scale: float) -> None:
        super().__init__()
        assert attn_dim % dim_heads == 0, "dim_model must be divisible by dim_heads"

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

    def forward(self, x: Tensor) -> Tensor:
        # calculate query, key, values for all heads in batch
        x = self.c_attn(x)

        # split into qkv and heads
        q, k, v = rearrange(x, "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads", n=3, dim_heads=self.dim_heads)

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
        attn_scale: float,
        mlp_dim_mult: int = 4,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.attn = SelfAttention(dim_model, attn_dim, dim_heads, attn_scale)
        self.ln_2 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.mlp = MLP(dim_model, mlp_dim_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MupActionExpert(nn.Module):
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
        attention_dim: int = 384,
        action_dim: int = 2,
        action_horizon: int = 6,
        number_high_level_command: int = 3,
        max_period: float = 10000.0,
        nb_layers: int = 12,
        dim_heads: int = 64,
        mlp_dim_mult: int = 4,
        init_std: float = 0.02,
        output_scale: float = 1.0,
        attn_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.number_high_level_command = number_high_level_command
        self.init_std = init_std
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.mlp_dim_mult = mlp_dim_mult
        self.nb_layers = nb_layers

        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            action_hidden_dim=embedding_dim,
            action_horizon=action_horizon,
            number_high_level_command=number_high_level_command,
            max_period=max_period,
        )

        self.transformer = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [
                        Block(
                            embedding_dim,
                            attention_dim,
                            dim_heads,
                            attn_scale,
                            mlp_dim_mult,
                        )
                        for _ in range(nb_layers)
                    ]
                ),
                "ln_f": nn.LayerNorm(embedding_dim, elementwise_affine=False),
            }
        )

        self.action_decoder = MuReadout(embedding_dim, action_dim, bias=False, output_mult=output_scale)

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
        return n_params

    def _init_c_proj_residual(self, module: nn.Module) -> None:
        """
        Apply special scaled init to the residual projections (attn & mlp), per GPT-2 paper

            > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            >   -- GPT-2 :: https://openai.com/blog/better-language-models/

        This is combined with the muP initilization that is scaling init_std by 1/sqrt(width).

        Note: for the MLP c_proj, the scaling is 1/sqrt(width * mlp_hidden_mult)
        """
        # times 2 because in a block there are 2 residual paths, attn & mlp

        depth_std = self.init_std * (2 * self.nb_layers)**-0.5

        for p_name, p in module.named_parameters():
            if p_name.endswith("c_proj.weight"):
                if hasattr(p, 'infshape'):
                    normal_(p, mean=0.0, std=depth_std)
                else:
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
        if isinstance(module, MuReadout):
            # https://arxiv.org/abs/2404.05728 | 4.2.6 SP Unembedding Initialization
            # A simple alternative to either initialization is a zeroinitialized unembedding projection [15],
            # which we found to perform similarly to the µP initialization and to also facilitate transfer
            module.weight.data.zero_()

        elif isinstance(module, nn.Linear):

            if hasattr(module.weight, "infshape"):
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
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

        elif isinstance(module, SelfAttention):
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

            self._init_c_proj_residual(module)

        elif isinstance(module, MLP):
            self._init_c_proj_residual(module)

    def forward(
        self, noisy_actions: Tensor, high_level_command: Tensor, t: Tensor
    ) -> Tensor:
        
        # noisy_actions: [Batch_Size, timesteps, Horizon_Steps, Action_Dim]
        action_embeds = self.action_encoder(
            actions=noisy_actions, high_level_command=high_level_command, diffusion_step=t
        )
        action_embeds = rearrange(action_embeds, "b t h d -> b (t h) d")
        
        for block in self.transformer.h:
            action_embeds = block(action_embeds)
        action_embeds = self.transformer.ln_f(action_embeds)
            
        denoised_actions = self.action_decoder(action_embeds)
        
        denoised_actions = rearrange(denoised_actions, "b (t h) d -> b t h d", t=noisy_actions.size(1))
        
        return denoised_actions


if __name__ == "__main__":
    import mup

    dit = MupActionExpert()

    # set base shapes
    base = mup.set_base_shapes(dit)

    bs = 3
    tens = torch.randn(bs, 6, 2)
    diffusion_step = torch.randint(0, 100, (bs,))
    high_level_command = torch.randint(0, 3, (bs,))
