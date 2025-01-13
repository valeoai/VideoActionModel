from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

from world_model.src.layers import TransformerLayer
from world_model.src.pos_embedding import precompute_freqs_cis


def print_model_info(parameters, width, depth, weight_tying, vocab_embed, spatial_pos_embed):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in parameters)
    vocab_params = vocab_embed.weight.numel()
    pos_params = spatial_pos_embed.weight.numel()

    def as_M(value):
        return f"{value / 1e6:.2f}M"

    if weight_tying:
        print(
            f"Width: {width} | Depth: {depth} | #Params: {as_M(n_params - vocab_params)} | of which vocab: {as_M(vocab_params)} | of which pos embed {as_M(pos_params)} "
        )
    else:
        print(
            f"Width: {width} | Depth: {depth} | #Params: {as_M(n_params)} | of which vocab: {as_M(vocab_params)} | of which pos embed {as_M(pos_params)} "
        )
    print(f"Codebook size: {tuple(vocab_embed.weight.shape)}")


class ARVideoNetwork(nn.Module):

    base_width = 128

    def __init__(
        self,
        width: int,
        depth: int,
        vocab_size: int,
        head_dim: int,
        ffn_expansion: int,
        nb_timesteps: int,
        nb_tokens_per_timestep: int,
        rope_theta: float,
        init_std: float = 0.02,
        weight_tying: bool = True,
    ) -> None:
        super().__init__()

        self.width = width
        self.depth = depth
        self.ffn_expansion = ffn_expansion
        self.nb_timesteps = nb_timesteps
        self.nb_tokens_per_timestep = nb_tokens_per_timestep
        self.init_std = init_std
        self.weight_tying = weight_tying

        self.spatial_pos_embed = nn.Embedding(nb_tokens_per_timestep, width)

        self.vocab_embed = nn.Embedding(vocab_size, width)
        self.layers = nn.Sequential(
            *(TransformerLayer(width, head_dim, ffn_expansion, layer_idx) for layer_idx in range(depth))
        )

        self.out_norm = nn.RMSNorm(width, elementwise_affine=False)
        self.out_proj = nn.Linear(width, vocab_size, bias=False)

        if self.weight_tying:
            # with weight tying when using torch.compile() some warnings get generated:
            # "UserWarning: functional_call was passed multiple values for tied weights.
            # This behavior is deprecated and will be an error in future versions"
            # not 100% sure what this is, so far seems to be harmless. TODO investigate
            self.out_proj.weight = self.vocab_embed.weight  # https://paperswithcode.com/method/weight-tying

        self.freqs_cis = precompute_freqs_cis(
            head_dim,
            nb_timesteps,
            rope_theta,
        )

        print_model_info(self.parameters(), width, depth, weight_tying, self.vocab_embed, self.spatial_pos_embed)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        In addition to muP we apply special scaled init to the residual projections (attn & mlp), per GPT-2 paper

            > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            >   -- GPT-2 :: https://openai.com/blog/better-language-models/

        This is combined with the muP initilization that is scaling init_std by sqrt(base_fan_in/fan_in).
        Note: there are 2 * self.depth # of residual layers because each transformer block has res attn and res mlp
        """

        # muP init
        # Note that if attention uses nn.Linear fanout is first dim
        # because nn.Linear applies y=xA.T+b
        # if using Conv1D as in hugginfaces transformers, fanout is last dim

        if isinstance(module, nn.Embedding):
            # Embeddings uses init_std directly
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)

        elif isinstance(module, TransformerLayer):

            # init all modules where base_fan_in = base_width
            base_fan_in = self.base_width
            fan_in = module.attn_out.weight.shape[1]
            std = self.init_std * (base_fan_in / fan_in) ** 0.5
            torch.nn.init.normal_(module.attn_qkv.weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.ffn_up.weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.ffn_gate.weight, mean=0.0, std=std)

            std = self.init_std * (2 * self.depth * base_fan_in / fan_in) ** 0.5
            torch.nn.init.normal_(module.attn_out.weight, mean=0.0, std=std)

            ### muP zero-initialized query projections
            # https://arxiv.org/abs/2404.05728
            # yield equal attention weights over all past timesteps at initialization
            # if attention uses nn.Linear change init in first dim
            # if using Conv1D, change init in last dim
            fanout, _ = module.attn_qkv.weight.shape
            assert fanout % 3 == 0  # assert matrix is used for query, key and value
            module.attn_qkv.weight.data[: fanout // 3, :] = 0

            # init all modules where base_fan_in != base_width
            base_fan_in = self.base_width * self.ffn_expansion
            fan_in = module.ffn_down.weight.shape[1]
            std = self.init_std * (2 * self.depth * base_fan_in / fan_in) ** 0.5
            torch.nn.init.normal_(module.ffn_down.weight, mean=0.0, std=std)

        elif module is self.out_proj and not self.weight_tying:
            # Zero-initialized unembedding projection
            # https://arxiv.org/abs/2404.05728
            module.weight.data[:] = 0

    def configure_optimizers(self, weight_decay, lr, *args, **kwargs):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        groups = defaultdict(list)

        ### Begin muP code ###
        for n, p in param_dict.items():
            if self.weight_tying and n.endswith("out_proj.weight"):
                continue

            if p.dim() < 2:
                # no weight decay, no lr scaling
                groups[(0.0, lr)].append(p)
                continue

            # for weights with weight decay we use decoupling
            # weight_decay/learning_rate
            # Note: only independent of peak LR, not of schedule

            if n.endswith("vocab_size.weight") or n.endswith("spatial_pos_embed.weight"):
                scaled_lr = lr / self.width**0.5
                groups[(weight_decay / scaled_lr, scaled_lr)].append(p)
                continue

            hidden_list = ["attn_qkv", "attn_out", "ffn_up", "ffn_gate", "ffn_down"]

            is_mup = False
            for mup_weight in hidden_list:
                if n.endswith(mup_weight + ".weight"):
                    is_mup = True

            if is_mup:
                fan_in = p.shape[1]
                base_factor = fan_in / self.width  # handle layers with width expansion
                scaled_lr = lr * self.base_width * base_factor / fan_in
                groups[(weight_decay / scaled_lr, scaled_lr)].append(p)
            else:
                groups[(weight_decay / lr, lr)].append(p)

        ### End muP code ###

        i = 0
        for (wd, lr), params in groups.items():
            num_params = sum(p.numel() for p in params)
            print(
                f"OPTIM GROUP {i} | wd: {wd:.2e} | lr {lr:.2e} | num parameter tensors: {len(params)}, with {num_params:,} parameters"
            )
            i += 1

        optim_groups = [{"params": params, "weight_decay": wd, "lr": lr} for (wd, lr), params in groups.items()]

        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=weight_decay, *args, **kwargs)

        return optimizer

    def forward(self, token_sequence, spatial_positions, temporal_positions, inference=False) -> Tensor:
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

        # compute spatial position embeddings
        spatial_pos_emb = self.spatial_pos_embed(spatial_positions)

        tok_emb = self.vocab_embed(token_sequence)

        hidden_states = tok_emb + spatial_pos_emb

        seqlen = hidden_states.size(1)
        attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool))
        attn_mask = attn_mask.type_as(hidden_states)

        self.freqs_cis = self.freqs_cis.to(token_sequence.device)
        freqs_cis = self.freqs_cis[temporal_positions]

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attn_mask=attn_mask,
                freqs_cis=freqs_cis,
            )

        logits = self.out_proj(hidden_states).float()
        logits = logits * self.base_width / self.width  # muP

        return logits


if __name__ == "__main__":
    # dummy params
    m = ARVideoNetwork(
        width=512,
        depth=8,
        vocab_size=2048,
        head_dim=128,
        ffn_expansion=4,
        nb_timesteps=3,
        nb_tokens_per_timestep=576,
        rope_theta=100,
        init_std=0.02,
    )

    # printing modules names
    for module_name, module in m.named_modules():
        print(module_name)

    # printing lr groups info
    optimizer = m.configure_optimizers(weight_decay=1e-5, lr=0.001)
