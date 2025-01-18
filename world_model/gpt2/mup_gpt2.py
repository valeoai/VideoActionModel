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

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange
from mup import MuReadout, MuSharedReadout, normal_
from torch import Tensor
from tqdm import tqdm

from world_model.gpt2.prepare_token_sequence import compute_position_indices


class KVCache(nn.Module):
    """
    Adapted from
    https://github.com/karpathy/nano-llama31/blob/06461cada7744a7da86a408f094549800b6bee3f/llama31.py#L141

    It is a simplified version as we assume that for each element of the batch there
    the same number of tokens.

    Also note that for the attention mask, we either have to recompute the full cache,
    so we use the causal mask, or we foward a single token so we don't need the mask.
    """

    def __init__(
        self, batch_size: int, seq_length: int, n_kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, n_kv_heads, seq_length, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.start_pos = 0

    def reset(self) -> None:
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.start_pos = 0

    def update(self, xk: Tensor, xv: Tensor) -> Tuple[Tensor, Tensor]:
        # changed from original implementation because shape in mup_GPT2 is (b nb_heads seq dim_head)
        seqlen = xk.size(2)
        self.cache_k[:, :, self.start_pos : self.start_pos + seqlen] = xk
        self.cache_v[:, :, self.start_pos : self.start_pos + seqlen] = xv
        xk = self.cache_k[:, :, : self.start_pos + seqlen]
        xv = self.cache_v[:, :, : self.start_pos + seqlen]
        self.start_pos += seqlen
        return xk, xv


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

    def __init__(self, dim_model: int, dim_heads: int, block_size: int, attn_scale: float) -> None:
        super().__init__()
        assert dim_model % dim_heads == 0, "dim_model must be divisible by dim_heads"

        self.dim_heads = dim_heads
        self.dim_model = dim_model
        self.nb_heads = dim_model // dim_heads

        ########### muP ###########
        self.attn_scale = attn_scale
        self.attn_score = nn.Identity()  # just for coordcheck
        self.query = nn.Identity()  # just for coordcheck
        self.key = nn.Identity()  # just for coordcheck
        self.value = nn.Identity()  # just for coordcheck

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(dim_model, 3 * dim_model, bias=False)

        # output projection
        self.c_proj = nn.Linear(dim_model, dim_model, bias=False)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:

        # calculate query, key, values for all heads in batch
        x = self.c_attn(x)

        # split into qkv and heads
        q, k, v = rearrange(x, "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads", n=3, dim_heads=self.dim_heads)

        # KV cache update
        if self.cache is not None:
            # update the KV cache with current KV and get all the previous KVs
            k, v = self.cache.update(k, v)

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
        dim_heads: int,
        block_size: int,
        attn_scale: float,
        mlp_dim_mult: int = 4,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.attn = CausalSelfAttention(dim_model, dim_heads, block_size, attn_scale)
        self.ln_2 = nn.LayerNorm(dim_model, elementwise_affine=False)
        self.mlp = MLP(dim_model, mlp_dim_mult)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class MupGPT2(nn.Module):
    """
    GPT2 implementation following the original formulation to be able to load existing pre-trained weights

    Args:
        embedding_dim: embedding dimension for the world model
        nb_layers: number of transformer blocks
        dim_heads: dimension of attention heads
        vocabulary_size: total number of vector embeddings in GPT's codebook
        nb_timesteps: maximum number of timesteps found in one sequence
        nb_tokens_per_timestep: number of tokens ithi each timestep
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        nb_layers: int = 12,
        dim_heads: int = 2,
        mlp_dim_mult: int = 4,
        vocabulary_size: int = 1104,
        nb_timesteps: int = 8,
        nb_tokens_per_timestep: int = 352,
        init_std: float = 0.02,
        output_tied: bool = True,
        output_scale: float = 1.0,
        attn_scale: float = 1.0,
    ) -> None:

        super().__init__()
        assert vocabulary_size is not None, "vocabulary_size must be provided"
        assert nb_tokens_per_timestep is not None, "nb_tokens_per_timestep must be provided"
        assert nb_timesteps is not None, "nb_timesteps must be provided"

        self.embedding_dim = embedding_dim
        self.init_std = init_std
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.mlp_dim_mult = mlp_dim_mult
        self.nb_layers = nb_layers
        self.output_tied = output_tied

        self.nb_timesteps = nb_timesteps
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        self.block_size = nb_timesteps * nb_tokens_per_timestep

        self.transformer = nn.ModuleDict(
            {
                "wie": nn.Embedding(vocabulary_size, embedding_dim),  # token embeddings
                "wse": nn.Embedding(nb_tokens_per_timestep, embedding_dim),  # spatial position embeddings
                "wte": nn.Embedding(nb_timesteps, embedding_dim),  # temporal position embeddings
                "h": nn.ModuleList(
                    [
                        Block(
                            embedding_dim,
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
            self.lm_head = MuSharedReadout(self.transformer.wie.weight, bias=False, output_mult=output_scale)
        else:
            self.lm_head = MuReadout(embedding_dim, vocabulary_size, bias=False, output_mult=output_scale)

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
            n_params -= self.transformer.wse.weight.numel()
            n_params -= self.transformer.wte.weight.numel()
        if not self.output_tied:
            n_params -= self.transformer.wie.weight.numel()
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
        depth_std = self.init_std * (2 * self.nb_layers) ** -0.5

        for p_name, p in module.named_parameters():
            if p_name.endswith(".c_proj.weight"):
                if hasattr(p, "infshape"):
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
        if isinstance(module, MuReadout) and not self.output_tied:
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

            self._init_c_proj_residual(module)

        elif isinstance(module, MLP):
            self._init_c_proj_residual(module)

    def forward(
        self,
        token_sequence: Tensor,
        spatial_positions: Tensor,
        temporal_positions: Tensor,
        inference: bool = False,
        use_kv_cache: bool = False,
    ) -> Tensor:
        """
        Args:
            token_sequence: A tensor of interleaved visual and action tokens.
            spatial_positions: A tensor indicating the spatial position of each token in the sequence.
                example: [0,1,2,3,0,1,2,3]
            temporal_positions: A tensor indicating the temporal position of each token in the sequence.
                example: [0,0,0,0,1,1,1,1]
        """
        assert (
            spatial_positions.max() < self.nb_tokens_per_timestep
        ), f"spatial_positions.max()={spatial_positions.max()} >= self.nb_tokens_per_timestep={self.nb_tokens_per_timestep}"
        assert (
            temporal_positions.max() < self.nb_timesteps
        ), f"temporal_positions.max()={temporal_positions.max()} >= self.nb_timesteps={self.nb_timesteps}"

        # compute spatio-temporal position embeddings
        spatial_pos_emb = self.transformer.wse(spatial_positions)
        temporal_pos_emb = self.transformer.wte(temporal_positions)

        tok_emb = self.transformer.wie(token_sequence)

        x = tok_emb + temporal_pos_emb + spatial_pos_emb

        seqlen = token_sequence.size(1)
        if inference and use_kv_cache:
            attn_mask = None
            assert seqlen == 1, "inference with KV cache only supports single token forward"
        else:
            attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=token_sequence.device, dtype=torch.bool))

        # forward world embeddings to the transformer
        for block in self.transformer.h:
            x = block(x, attn_mask)
        emb_out = self.transformer.ln_f(x)

        if not inference:
            img_logits = self.lm_head(emb_out)
        else:
            img_logits = self.lm_head(emb_out[:, [-1], :])  # note: using list [-1] to preserve the time dim

        return img_logits

    def _setup_kv_cache(self, batch_size: int) -> None:
        """
        Sets up key-value caching for transformer attention layers.

        KV Cache:
        - Stores the Key and Value projections for each token
        - Avoids recomputing these for previous tokens during generation
        - Cache size matches max_context_size
        - Must be reset when context is rolled
        """
        for block in self.transformer.h:
            layer = block.attn
            cache = KVCache(
                batch_size=batch_size,
                seq_length=self.block_size,
                n_kv_heads=layer.nb_heads,
                head_dim=layer.dim_heads,
                dtype=layer.c_attn.weight.dtype,
                device=layer.c_attn.weight.device,
            )
            layer.cache = cache

    def _reset_kv_cache(self) -> None:
        """Reset the key-value cache."""
        for block in self.transformer.h:
            block.attn.cache.reset()

    def _clear_kv_cache(self) -> None:
        """Clear the key-value cache."""
        for block in self.transformer.h:
            block.attn.cache = None

    def _sample_next_token(self, logits: Tensor, temperature: float, topk_sampler: int) -> Tensor:
        """
        Sample the next token from the logits using top-k sampling.

        Args:
            logits: The logits from the model.
            temperature: The temperature for sampling.
            topk_sampler: The number of top-k tokens to sample from.

        Returns:
            The sampled tokens
        """
        logits = logits[:, -1, :] / temperature
        tokens_probs = torch.softmax(logits, dim=-1)
        topk_probs, topk_tokens = torch.topk(tokens_probs, topk_sampler, dim=-1, sorted=False)
        topk_renormalized_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        # Sample from the top k normalized probabilities
        next_token_idx = torch.multinomial(topk_renormalized_probs, num_samples=1)
        next_token = torch.gather(topk_tokens, -1, next_token_idx)
        return next_token

    def _process_generated_frame(self, context: Tensor, height: int, width: int) -> Tensor:
        """Extract and process the latest generated frame."""
        frame = context[:, -self.nb_tokens_per_timestep :]
        frame = rearrange(frame, "b (h w) -> b h w", h=height, w=width)
        return frame

    def forward_inference(
        self,
        number_of_future_frames: int,
        burnin_visual_tokens: Tensor,
        temperature: float = 1.0,
        topk_sampler: int = 1,
        use_kv_cache: bool = False,
        verbose: int | bool = 0,
    ) -> Tensor:
        verbose = int(verbose)

        bs, _, height, width = burnin_visual_tokens.shape
        context = rearrange(burnin_visual_tokens, "b t h w -> b (t h w)")

        def _count_nb_frames(context: Tensor) -> int:
            return context.size(1) // (height * width)

        # we get positions for the max context size we are allowed
        positions = compute_position_indices(bs, self.nb_timesteps, height, width)

        # cut the context to the maximum context size
        context = context[:, -self.block_size :]
        spatial_positions = positions["spatial_positions"]
        temporal_positions = positions["temporal_positions"]

        generated_frames = torch.zeros(
            bs, number_of_future_frames, height, width, device=burnin_visual_tokens.device, dtype=burnin_visual_tokens.dtype
        )

        if use_kv_cache:
            self._setup_kv_cache(bs)
            kv_cache_was_reset = True

        for frame_idx in tqdm(
            range(number_of_future_frames), f"Generating {number_of_future_frames} frames", disable=verbose < 1
        ):

            # We should always have at most self.nb_timesteps - 1 frames in the context
            # to leave space for the generated frame
            if _count_nb_frames(context) > self.nb_timesteps - 1:
                context = context[:, : (self.nb_timesteps - 1) * self.nb_tokens_per_timestep]
                # because we use learned positional embeddings, we can not keep the context
                # in cache and simply roll it. We need to recompute the whole cache.
                self._reset_kv_cache()
                kv_cache_was_reset = True

            for _ in tqdm(
                range(self.nb_tokens_per_timestep),
                f"AR generation of {self.nb_tokens_per_timestep} tokens",
                disable=verbose < 2,
                leave=False,
                position=1,
            ):
                # Get next token
                if use_kv_cache and not kv_cache_was_reset:
                    tmp_ctx = context[:, -1:]
                    tmp_spatial_positions = spatial_positions[:, -1:]
                    tmp_temporal_positions = temporal_positions[:, -1:]
                else:
                    tokens_in_context = context.size(1)
                    tmp_ctx = context
                    tmp_spatial_positions = spatial_positions[:, :tokens_in_context]
                    tmp_temporal_positions = temporal_positions[:, :tokens_in_context]

                logits = self.forward(
                    tmp_ctx,
                    tmp_spatial_positions,
                    tmp_temporal_positions,
                    inference=True,
                    use_kv_cache=use_kv_cache and not kv_cache_was_reset,
                )
                next_tokens = self._sample_next_token(logits, temperature, topk_sampler)
                context = torch.cat([context, next_tokens], dim=1)
                kv_cache_was_reset = False

            generated_frames[:, frame_idx] = self._process_generated_frame(context, height, width)

        if use_kv_cache:
            self._clear_kv_cache()
        return generated_frames


if __name__ == "__main__":
    import mup

    height, width = 8, 12

    model = MupGPT2(
        embedding_dim=128, nb_layers=4, nb_tokens_per_timestep=height * width, nb_timesteps=8, vocabulary_size=1024
    )
    mup.set_base_shapes(model, None)

    visual_tokens = torch.randint(0, 1024, (1, 4, 8, 12), dtype=torch.long)

    generated_frames = model.forward_inference(2, visual_tokens, temperature=1.0, topk_sampler=3, use_kv_cache=True, verbose=2)
