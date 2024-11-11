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
import torch
import torch.nn as nn

from einops import rearrange


from mup import MuReadout, MuSharedReadout, normal_


class MLP(nn.Module):

    def __init__(self, dim_model, bias, dropout, hidden_dim_mult=4):
        super().__init__()
        self.c_fc = nn.Linear(dim_model, hidden_dim_mult * dim_model, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim_mult * dim_model, dim_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):

    def __init__(self, dim_model, nb_heads, bias, dropout, block_size, attn_scale):
        super().__init__()
        assert dim_model % nb_heads == 0, "dim_model must be divisible by nb_heads"

        self.nb_heads = nb_heads
        self.dim_model = dim_model

        ########### muP ###########
        self.attn_scale = attn_scale
        self.attn_score = nn.Identity()  # just for coordcheck
        self.query = nn.Identity()  # just for coordcheck
        self.key = nn.Identity()  # just for coordcheck
        self.value = nn.Identity()  # just for coordcheck

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(dim_model, 3 * dim_model, bias=bias)

        # output projection
        self.c_proj = nn.Linear(dim_model, dim_model, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

        # will be KVCache object managed by inference context manager
        self.cache = None

    def forward(self, x, attn_mask, start_pos: int = None):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (dim_model)

        # calculate query, key, values for all heads in batch
        x = self.c_attn(x)

        # split into qkv and heads
        q, k, v = rearrange(x, "b seq (n nb_heads dim_head) -> n b nb_heads seq dim_head", n=3, nb_heads=self.nb_heads)

        # KV cache update
        if self.cache is not None:
            assert isinstance(start_pos, int), "start_pos must be an integer"
            # update the KV cache with current KV and get all the previous KVs
            k, v = self.cache.update(start_pos, k, v)

        ### muP: just for coord check (debug)
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        ### muP: attention scaling 1/dim_head instead of 1/sqrt(dim_head)
        attn_scaling = self.attn_scale / v.size(-1)

        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
            scale=attn_scaling,  # muP: attention scaling
        )

        y = rearrange(y, "b nb_heads seq dim_head -> b seq (nb_heads dim_head)")  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


class Block(nn.Module):

    def __init__(self, dim_model, nb_heads, bias, dropout, block_size, attn_scale, learnable_gains=False, mlp_dim_mult=4):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim_model, elementwise_affine=learnable_gains)
        self.attn = CausalSelfAttention(dim_model, nb_heads, bias, dropout, block_size, attn_scale)
        self.ln_2 = nn.LayerNorm(dim_model, elementwise_affine=learnable_gains)
        self.mlp = MLP(dim_model, bias, dropout, mlp_dim_mult)

    def forward(self, x, attn_mask, start_pos=None):
        x = x + self.attn(self.ln_1(x), attn_mask, start_pos=start_pos)
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
        bias: True: bias in Linears. False: a bit better and faster
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        nb_layers: int = 12,
        nb_heads: int = 16,
        mlp_dim_mult: int = 4,
        vocabulary_size: int = 1104,
        nb_timesteps: int = 8,
        nb_tokens_per_timestep: int = 352,
        multiple_tokens_inference: bool = False,
        dropout_rate: float = 0.0,
        init_std: float = 0.02,
        bias: bool = True,
        output_tied: bool = True,
        output_scale: float = 1.0,
        attn_scale: float = 1.0,
        learnable_gains: bool = False
    ) -> None:

        super().__init__()
        assert vocabulary_size is not None, "vocabulary_size must be provided"
        assert nb_tokens_per_timestep is not None, "nb_tokens_per_timestep must be provided"
        assert nb_timesteps is not None, "nb_timesteps must be provided"

        self.embedding_dim = embedding_dim
        self.bias = bias
        self.init_std = init_std
        self.attn_scale = attn_scale
        self.output_scale = output_scale
        self.mlp_dim_mult = mlp_dim_mult
        self.nb_layers = nb_layers
        self.output_tied = output_tied
        self.vocab_size = vocabulary_size

        self.nb_timesteps = nb_timesteps
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        self.block_size = nb_timesteps * nb_tokens_per_timestep
        self.multiple_tokens_inference = multiple_tokens_inference

        self.transformer = nn.ModuleDict(dict(
            wie=nn.Embedding(vocabulary_size, embedding_dim),         # token embeddings
            wse=nn.Embedding(nb_tokens_per_timestep, embedding_dim),  # spatial position embeddings
            wte=nn.Embedding(nb_timesteps, embedding_dim),            # temporal position embeddings
            drop=nn.Dropout(dropout_rate),
            h=nn.ModuleList([
                Block(embedding_dim, nb_heads, bias, dropout_rate, self.block_size, attn_scale, learnable_gains, mlp_dim_mult)
                for _ in range(nb_layers)
            ]),
            ln_f=nn.LayerNorm(embedding_dim, elementwise_affine=learnable_gains),
        ))

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
        if not self.output_tied:
            n_params -= self.transformer.wie.weight.numel()
        return n_params

    def _init_c_proj_residual(self, module, is_mlp):
        '''
        Apply special scaled init to the residual projections (attn & mlp), per GPT-2 paper

            > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            >   -- GPT-2 :: https://openai.com/blog/better-language-models/

        This is combined with the muP initilization that is scaling init_std by 1/sqrt(width).

        Note: for the MLP c_proj, the scaling is 1/sqrt(width * mlp_hidden_mult)
        '''
        # times 2 because in a block there are 2 residual paths, attn & mlp
        scaling = 2 * self.nb_layers * self.embedding_dim

        if is_mlp:
            scaling *= self.mlp_dim_mult

        depth_std = self.init_std * scaling**-0.5

        for p_name, p in module.named_parameters():
            if p_name.endswith('c_proj.weight'):
                p.data.normal_(mean=0.0, std=depth_std)

    def _init_weights(self, module):
        '''
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
            Dropout(p=0.0, inplace=False)
            ------------------------------
            MLP(
            (c_fc): Linear(in_features=128, out_features=512, bias=False)
            (gelu): GELU(approximate='none')
            (c_proj): Linear(in_features=512, out_features=128, bias=False)
            (dropout): Dropout(p=0.0, inplace=False)
            )

        This means that specific initialization to MLP using isinstance(module, MLP)
        will override the initialization from isinstance(module, nn.Linear)

        We use this fact apply specific initialization rules in some modules.
        For example the init scaling of the MLP hidden nn.Linear layer is different
        than other nn.Linear layers.
        '''

        # MuReadout zero init
        if isinstance(module, MuReadout) and not self.output_tied:
            # https://arxiv.org/abs/2404.05728 | 4.2.6 SP Unembedding Initialization
            # A simple alternative to either initialization is a zeroinitialized unembedding projection [15],
            # which we found to perform similarly to the µP initialization and to also facilitate transfer
            module.weight.data.zero_()

        elif isinstance(module, nn.Linear):

            if hasattr(module.weight, 'infshape'):
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
            module.c_attn.weight.data[:fanout // 3, :] = 0

            self._init_c_proj_residual(module, is_mlp=False)

        elif isinstance(module, MLP):
            self._init_c_proj_residual(module, is_mlp=True)

    def forward(
        self,
        token_sequence,
        spatial_positions,
        temporal_positions,
        inference=False,
        start_pos=-1,
    ):
        """
        Args:
            token_sequence: A tensor of interleaved visual and action tokens.
            spatial_positions: A tensor indicating the spatial position of each token in the sequence.
                example: [0,1,2,3,0,1,2,3]
            temporal_positions: A tensor indicating the temporal position of each token in the sequence.
                example: [0,0,0,0,1,1,1,1]
        """

        assert spatial_positions.max() < self.nb_tokens_per_timestep, f"spatial_positions.max()={spatial_positions.max()} >= self.nb_tokens_per_timestep={self.nb_tokens_per_timestep}"
        assert temporal_positions.max() < self.nb_timesteps, f"temporal_positions.max()={temporal_positions.max()} >= self.nb_timesteps={self.nb_timesteps}"

        # compute spatio-temporal position embeddings
        spatial_pos_emb = self.transformer.wse(spatial_positions)
        temporal_pos_emb = self.transformer.wte(temporal_positions)

        tok_emb = self.transformer.wie(token_sequence)

        emb_in = tok_emb + temporal_pos_emb + spatial_pos_emb

        seqlen = token_sequence.size(1)
        if inference and start_pos != -1:
            attn_mask = None
            if seqlen > 1:
                attn_mask = torch.full((seqlen, seqlen), float("-inf"), device=token_sequence.device)
                attn_mask = torch.triu(attn_mask, diagonal=1)
                # When performing key-value caching, we compute the attention scores
                # only for the new sequence. Thus, the matrix of scores is of size
                # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
                # j > cache_len + i, since row i corresponds to token cache_len + i.
                attn_mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=token_sequence.device), attn_mask]
                ).type_as(emb_in)
        else:
            if self.multiple_tokens_inference:
                # For the attention mask we want:
                # 1. Do not attend to future frames
                # 2. Attend to all spatial tokens in the same frame (this makes it not entirely causal)
                attn_mask = temporal_positions.unsqueeze(1) <= temporal_positions.unsqueeze(2)
                attn_mask.unsqueeze_(1)  # this is for the nb_heads dimension
            else:
                # Full causal mask
                attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=token_sequence.device, dtype=torch.bool))

        # forward world embeddings to the transformer
        x = self.transformer.drop(emb_in)
        for block in self.transformer.h:
            x = block(x, attn_mask, start_pos=start_pos)
        emb_out = self.transformer.ln_f(x)

        if not inference:
            img_logits = self.lm_head(emb_out)
        else:
            if self.multiple_tokens_inference:
                # at inference time, we only we keep the prediction for all the tokens of the last timestep
                img_logits = self.lm_head(emb_out[:, -self.nb_tokens_per_timestep:])
            else:
                # at inference time, we only we keep the prediction for the last timestep
                img_logits = self.lm_head(emb_out[:, [-1]])

        return img_logits


if __name__ == '__main__':
    # temporary test code
    import mup

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    voc_size = 1024
    nb_tokens_per_timestep = 8 * 16

    gpt2 = MuGPT2(
        embedding_dim=16,
        nb_layers=3,
        nb_heads=2,
        mlp_dim_mult=2,
        vocabulary_size=voc_size,
        nb_timesteps=8,
        nb_tokens_per_timestep=nb_tokens_per_timestep,
        multiple_tokens_inference=True,
        bias=False,
        output_tied=True,
    )
    mup.set_base_shapes(gpt2, base=None)
    gpt2.to(device)

    batch_size = 3
    number_of_frames = 5
    seqlen = nb_tokens_per_timestep * number_of_frames

    x = torch.randint(0, voc_size, (batch_size, seqlen))
    spatial_positions = torch.arange(nb_tokens_per_timestep).repeat(number_of_frames)
    temporal_positions = torch.arange(number_of_frames).repeat_interleave(nb_tokens_per_timestep)

    spatial_positions = spatial_positions.unsqueeze(0).repeat(batch_size, 1)
    temporal_positions = temporal_positions.unsqueeze(0).repeat(batch_size, 1)

    x = x.to(device)
    spatial_positions = spatial_positions.to(device)
    temporal_positions = temporal_positions.to(device)

    img_logits = gpt2(x, spatial_positions, temporal_positions)
    print(img_logits.shape)

    img_inf_logits = gpt2(x, spatial_positions, temporal_positions, inference=True)
    print(img_inf_logits.shape)
