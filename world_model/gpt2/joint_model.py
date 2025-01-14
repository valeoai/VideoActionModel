from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch import BoolTensor, FloatTensor, LongTensor

from world_model.gpt2.mup_action_expert import Block as ActionBlock
from world_model.gpt2.mup_action_expert import MupActionExpert
from world_model.gpt2.mup_action_expert import SelfAttention as ActionAttention
from world_model.gpt2.mup_gpt2 import Block as GPTBlock
from world_model.gpt2.mup_gpt2 import CausalSelfAttention as GPTAttention
from world_model.gpt2.mup_gpt2 import MupGPT2
from world_model.gpt2.prepare_token_sequence import prepare_token_sequence

EmbedsDict = Dict[str, FloatTensor]
InputsDict = Dict[str, FloatTensor | LongTensor]


class JointModel(nn.Module):

    def __init__(
        self,
        gpt: MupGPT2,
        action_expert: MupActionExpert,
    ) -> None:
        super().__init__()
        self.gpt = gpt
        self.action_expert = action_expert

        # Archi parameters
        self.num_hidden_layers = len(self.gpt.transformer.h)

    def _visual_tokens_to_embeds(self, visual_tokens: LongTensor) -> FloatTensor:
        sequence_data = prepare_token_sequence(visual_tokens)
        spatial_pos_emb = self.gpt.transformer.wse(sequence_data["spatial_positions"])
        temporal_pos_emb = self.gpt.transformer.wte(sequence_data["temporal_positions"])
        tok_emb = self.gpt.transformer.wie(sequence_data["token_sequence"])
        visual_embeds = tok_emb + temporal_pos_emb + spatial_pos_emb
        return visual_embeds

    def _noisy_action_to_embeds(self, noisy_action: FloatTensor, high_level_command: LongTensor, t: FloatTensor) -> FloatTensor:
        # noisy_action: [Batch_Size, timesteps, Horizon_Steps, Action_Dim]
        action_embeds = self.action_expert.action_encoder(
            actions=noisy_action, high_level_command=high_level_command, diffusion_step=t
        )
        action_embeds = rearrange(action_embeds, "b t h d -> b (t h) d")
        return action_embeds

    def _forward_mutual_attention(
        self,
        attn_mask: BoolTensor,
        visual_attn_input: FloatTensor,
        action_attn_input: FloatTensor,
        gpt_attention: GPTAttention,
        action_attention: ActionAttention,
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
            action_attention.c_attn(action_attn_input),
            "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
            n=3,
            dim_heads=action_attention.dim_heads,
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
        action_y = action_attention.c_proj(action_y)
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
        action_block: ActionBlock = self.action_expert.transformer.h[layer_idx]
        gpt_attention: GPTAttention = gpt_block.attn
        action_attention: ActionAttention = action_block.attn

        visual_attn_input = gpt_block.ln_1(visual_embeds)
        action_attn_input = action_block.ln_1(action_embeds)

        visual_attn_output, action_attn_output = self._forward_mutual_attention(
            attention_mask, visual_attn_input, action_attn_input, gpt_attention, action_attention
        )

        visual_embeds = visual_embeds + visual_attn_output
        action_embeds = action_embeds + action_attn_output

        visual_embeds = visual_embeds + gpt_block.mlp(gpt_block.ln_2(visual_embeds))
        action_embeds = action_embeds + action_block.mlp(action_block.ln_2(action_embeds))

        return {"visual_embeds": visual_embeds, "action_embeds": action_embeds}

    def forward(
        self,
        attention_mask: BoolTensor,
        inputs_all: InputsDict,
    ) -> FloatTensor:
        # Get visual embeddings
        visual_embeds = self._visual_tokens_to_embeds(inputs_all["visual_tokens"])

        # Get action embeddings
        action_embeds = self._noisy_action_to_embeds(
            inputs_all["noisy_actions"], inputs_all["high_level_command"], inputs_all["diffusion_step"]
        )

        embeds_all = {"visual_embeds": visual_embeds, "action_embeds": action_embeds}
        for layer_idx in range(self.num_hidden_layers):
            embeds_all = self._forward_single_layer(attention_mask, embeds_all, layer_idx)

        action_embeds = self.action_expert.transformer.ln_f(embeds_all["action_embeds"])

        # [Batch_Size, Horizon_Steps, Action_Dim]
        denoised_actions = self.action_expert.action_decoder(action_embeds)

        return {"actions": denoised_actions, "actions_embeds": action_embeds}


if __name__ == "__main__":

    height, width = 8, 16

    gpt_config = {
        "embedding_dim": 128,
        "nb_layers": 12,
        "dim_heads": 16,
        "vocabulary_size": 1500,
        "nb_timesteps": 8,
        "nb_tokens_per_timestep": height * width,
    }
    action_expert_config = {
        "embedding_dim": 64,
        "attention_dim": 128,
        "action_dim": 2,
        "action_horizon": 6,
        "number_high_level_command": 3,
        "dim_heads": 16,
        "nb_layers": 12,
    }

    gpt = MupGPT2(**gpt_config)
    action_expert = MupActionExpert(**action_expert_config)

    joint_model = JointModel(gpt, action_expert)

    batch_size = 3

    attn_mask = None
    inputs_all = {
        "visual_tokens": torch.randint(
            0, gpt_config["vocabulary_size"], (batch_size, gpt_config["nb_timesteps"], height, width)
        ),
        "action_embeds": torch.randn(
            batch_size,
            gpt_config["nb_timesteps"] * action_expert_config["action_horizon"],
            action_expert_config["embedding_dim"],
        ),
    }

    action_embeds = joint_model(attn_mask, inputs_all)

    print("Output shape:", action_embeds.shape)
