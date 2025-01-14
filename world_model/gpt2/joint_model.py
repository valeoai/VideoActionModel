from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import BoolTensor, FloatTensor, LongTensor

from world_model.gpt2.mup_dit import Block as DiTBlock
from world_model.gpt2.mup_dit import MupDiT
from world_model.gpt2.mup_dit import SelfAttention as DiTAttention
from world_model.gpt2.mup_gpt2 import Block as GPTBlock
from world_model.gpt2.mup_gpt2 import CausalSelfAttention as GPTAttention
from world_model.gpt2.mup_gpt2 import MupGPT2
from world_model.gpt2.prepare_token_sequence import prepare_token_sequence

EmbedsDict = Dict[str, FloatTensor]
InputsDict = Dict[str, FloatTensor | LongTensor]


class JointModel(nn.Module):

    def __init__(
        self,
        gpt_config: OmegaConf,
        dit_config: OmegaConf,
    ) -> None:
        super().__init__()
        self.gpt: MupGPT2 = instantiate(gpt_config)
        self.dit: MupDiT = instantiate(dit_config)

        # Archi parameters
        self.num_hidden_layers = len(self.gpt.transformer.h)

    def _visual_tokens_to_embeds(self, visual_tokens: LongTensor) -> FloatTensor:
        sequence_data = prepare_token_sequence(visual_tokens)
        spatial_pos_emb = self.gpt.transformer.wse(sequence_data["spatial_positions"])
        temporal_pos_emb = self.gpt.transformer.wte(sequence_data["temporal_positions"])
        tok_emb = self.gpt.transformer.wie(sequence_data["token_sequence"])
        visual_embeds = tok_emb + temporal_pos_emb + spatial_pos_emb
        return visual_embeds

    def _forward_mutual_attention(
        self,
        attn_mask: BoolTensor,
        visual_attn_input: FloatTensor,
        action_attn_input: FloatTensor,
        gpt_attention: GPTAttention,
        dit_attention: DiTAttention,
    ) -> Tuple[FloatTensor, FloatTensor]:
        # split into qkv and heads
        visual_q, visual_k, visual_v = rearrange(
            gpt_attention.c_attn(visual_attn_input),
            "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
            n=3,
            dim_heads=gpt_attention.dim_heads,
        )
        ### muP: attention scaling 1/dim_heads instead of 1/sqrt(dim_heads)
        attn_scaling = gpt_attention.attn_scale / visual_v.size(-1)

        action_q, action_k, action_v = rearrange(
            dit_attention.c_attn(action_attn_input),
            "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
            n=3,
            dim_heads=dit_attention.dim_heads,
        )

        # attention
        q = torch.cat([visual_q, action_q], dim=-2)
        k = torch.cat([visual_k, action_k], dim=-2)
        v = torch.cat([visual_v, action_v], dim=-2)

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

        visual_y, action_y = torch.split(y, [visual_q.size(-2), action_q.size(-2)], dim=1)
        # output projection
        visual_y = gpt_attention.c_proj(visual_y)
        action_y = dit_attention.c_proj(action_y)
        return visual_y, action_y

    def _forward_single_layer(
        self,
        attention_mask: BoolTensor,
        embeds_all: EmbedsDict,
        layer_idx: int,
    ) -> EmbedsDict:
        visual_embeds = embeds_all["visual_embeds"]
        action_embeds = embeds_all["action_embeds"]

        gpt_block: GPTBlock = self.gpt.transformer.h[layer_idx]
        dit_block: DiTBlock = self.dit.transformer.h[layer_idx]
        gpt_attention: GPTAttention = gpt_block.attn
        dit_attention: DiTAttention = dit_block.attn

        visual_attn_input = gpt_block.ln_1(visual_embeds)
        action_attn_input = dit_block.ln_1(action_embeds)

        visual_attn_output, action_attn_output = self._forward_mutual_attention(
            attention_mask, visual_attn_input, action_attn_input, gpt_attention, dit_attention
        )

        visual_embeds = visual_embeds + visual_attn_output
        action_embeds = action_embeds + action_attn_output

        visual_embeds = visual_embeds + gpt_block.mlp(gpt_block.ln_2(visual_embeds))
        action_embeds = action_embeds + dit_block.mlp(dit_block.ln_2(action_embeds))

        return {"visual_embeds": visual_embeds, "action_embeds": action_embeds}

    def forward(
        self,
        attention_mask: BoolTensor,
        inputs_all: InputsDict,
    ) -> FloatTensor:
        visual_embeds = self._visual_tokens_to_embeds(inputs_all["visual_tokens"])

        embeds_all = {"visual_embeds": visual_embeds, "action_embeds": inputs_all["action_embeds"]}
        for layer_idx in range(self.num_hidden_layers):
            embeds_all = self._forward_single_layer(attention_mask, embeds_all, layer_idx)

        embeds_all["visual_embeds"] = self.gpt.transformer.ln_f(embeds_all["visual_embeds"])
        embeds_all["action_embeds"] = self.dit.transformer.ln_f(embeds_all["action_embeds"])

        return embeds_all


if __name__ == "__main__":
    height, width = 8, 16

    gpt_config = {
        "_target_": "world_model.gpt2.mup_gpt2.MupGPT2",
        "embedding_dim": 128,
        "nb_layers": 12,
        "dim_heads": 16,
        "vocabulary_size": 1500,
        "nb_timesteps": 8,
        "nb_tokens_per_timestep": height * width,
    }
    dit_config = {
        "_target_": "world_model.gpt2.mup_dit.MupDiT",
        "embedding_dim": 64,
        "attention_dim": 128,
        "dim_heads": 16,
        "nb_layers": 12,
    }

    gpt_config = OmegaConf.create(gpt_config)
    dit_config = OmegaConf.create(dit_config)

    joint_model = JointModel(gpt_config, dit_config)

    batch_size = 3
    action_horizon = 6

    attn_mask = None
    inputs_all = {
        "visual_tokens": torch.randint(
            0, gpt_config.vocabulary_size, (batch_size, gpt_config.nb_timesteps, height, width)
        ),
        "action_embeds": torch.randn(batch_size, gpt_config.nb_timesteps * action_horizon, dit_config.embedding_dim),
    }

    output = joint_model(attn_mask, inputs_all)

    print("Output shape:", output["visual_embeds"].shape, output["action_embeds"].shape)
