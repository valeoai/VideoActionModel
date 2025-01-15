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


class KVCache:

    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.reset()

    def reset(self) -> None:
        self.k_cache = [None for _ in range(self.num_layers)]
        self.v_cache = [None for _ in range(self.num_layers)]
        self._filled = [False for _ in range(self.num_layers)]

    @property
    def filled(self) -> bool:
        return all(self._filled)

    def update(self, k_cache: FloatTensor, v_cache: FloatTensor, layer_idx: int) -> None:
        self.k_cache[layer_idx] = k_cache
        self.v_cache[layer_idx] = v_cache
        self._filled[layer_idx] = True

    def get(self, layer_idx: int) -> Tuple[FloatTensor, FloatTensor]:
        assert self._filled[layer_idx], "Cache not filled"
        return self.k_cache[layer_idx], self.v_cache[layer_idx]


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

        # Cache for key and value
        self.kv_cache = None

    def init_kv_cache(self) -> None:
        self.kv_cache = KVCache(self.num_hidden_layers)

    def cleanup_kv_cache(self) -> None:
        self.kv_cache.reset()

    @property
    def use_kv_cache(self) -> None:
        return self.kv_cache is not None and self.kv_cache.filled

    def _visual_tokens_to_embeds(self, visual_tokens: LongTensor) -> FloatTensor:
        sequence_data = prepare_token_sequence(visual_tokens)
        spatial_pos_emb = self.gpt.transformer.wse(sequence_data["spatial_positions"])
        temporal_pos_emb = self.gpt.transformer.wte(sequence_data["temporal_positions"])
        tok_emb = self.gpt.transformer.wie(sequence_data["token_sequence"])
        visual_embeds = tok_emb + temporal_pos_emb + spatial_pos_emb
        return visual_embeds

    def _noisy_action_to_embeds(
        self, noisy_action: FloatTensor, high_level_command: LongTensor, t: FloatTensor
    ) -> FloatTensor:
        # noisy_action: [Batch_Size, timesteps, Horizon_Steps, Action_Dim]
        action_embeds = self.action_expert.action_encoder(
            actions=noisy_action, high_level_command=high_level_command, diffusion_step=t
        )
        action_embeds = rearrange(action_embeds, "b t h d -> b (t h) d")
        return action_embeds

    def _forward_mutual_attention(
        self,
        attn_mask: BoolTensor,
        visual_attn_input: FloatTensor | None,
        action_attn_input: FloatTensor,
        gpt_attention: GPTAttention,
        action_attention: ActionAttention,
        layer_idx: int,
    ) -> Tuple[FloatTensor, FloatTensor]:
        # split into qkv and heads
        if self.use_kv_cache:
            visual_k, visual_v = self.kv_cache.get(layer_idx)
        else:
            visual_q, visual_k, visual_v = rearrange(
                gpt_attention.c_attn(visual_attn_input),
                "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
                n=3,
                dim_heads=gpt_attention.dim_heads,
            )
            if self.kv_cache is not None:
                self.kv_cache.update(visual_k, visual_v, layer_idx)

        action_q, action_k, action_v = rearrange(
            action_attention.c_attn(action_attn_input),
            "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
            n=3,
            dim_heads=action_attention.dim_heads,
        )
        ### muP: attention scaling 1/dim_heads instead of 1/sqrt(dim_heads)
        attn_scaling = action_attention.attn_scale / action_v.size(-1)

        # attention
        if self.use_kv_cache:
            q = action_q
        else:
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

        if self.use_kv_cache:
            action_y = y
        else:
            visual_y, action_y = torch.split(y, [visual_q.size(-2), action_q.size(-2)], dim=1)
            visual_y = gpt_attention.c_proj(visual_y)
        # output projection
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

        if not self.use_kv_cache:
            visual_attn_input = gpt_block.ln_1(visual_embeds)
        else:
            visual_attn_input = None
        action_attn_input = action_block.ln_1(action_embeds)

        visual_attn_output, action_attn_output = self._forward_mutual_attention(
            attention_mask, visual_attn_input, action_attn_input, gpt_attention, action_attention, layer_idx
        )

        if not self.use_kv_cache:
            visual_embeds = visual_embeds + visual_attn_output
            visual_embeds = visual_embeds + gpt_block.mlp(gpt_block.ln_2(visual_embeds))

        action_embeds = action_embeds + action_attn_output
        action_embeds = action_embeds + action_block.mlp(action_block.ln_2(action_embeds))

        return {"visual_embeds": visual_embeds, "action_embeds": action_embeds}

    def forward(
        self,
        attention_mask: BoolTensor,
        inputs_all: InputsDict,
    ) -> FloatTensor:
        # Get visual embeddings
        if not self.use_kv_cache:
            visual_embeds = self._visual_tokens_to_embeds(inputs_all["visual_tokens"])
        else:
            visual_embeds = None

        # Get action embeddings
        action_embeds = self._noisy_action_to_embeds(
            inputs_all["noisy_actions"], inputs_all["high_level_command"], inputs_all["diffusion_step"]
        )

        embeds_all = {"visual_embeds": visual_embeds, "action_embeds": action_embeds}
        for layer_idx in range(self.num_hidden_layers):
            embeds_all = self._forward_single_layer(attention_mask, embeds_all, layer_idx)

        action_embeds = self.action_expert.transformer.ln_f(embeds_all["action_embeds"])

        # [Batch_Size, context_length, Horizon_Steps, Action_Dim]
        denoised_actions = self.action_expert.action_decoder(action_embeds)

        denoised_actions = rearrange(denoised_actions, "b (t h) d -> b t h d", t=inputs_all["noisy_actions"].size(1))
        action_embeds = rearrange(action_embeds, "b (t h) d -> b t h d", t=inputs_all["noisy_actions"].size(1))
        return {"actions": denoised_actions, "actions_embeds": action_embeds}


if __name__ == "__main__":

    import mup

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
    mup.set_base_shapes(gpt, None)
    mup.set_base_shapes(action_expert, None)

    joint_model = JointModel(gpt, action_expert)

    batch_size = 3

    attn_mask = None
    inputs_all = {
        "visual_tokens": torch.randint(
            0, gpt_config["vocabulary_size"], (batch_size, gpt_config["nb_timesteps"], height, width)
        ),
        "noisy_actions": torch.randn(
            batch_size,
            gpt_config["nb_timesteps"],
            action_expert_config["action_horizon"],
            action_expert_config["action_dim"],
        ),
        "high_level_command": torch.randint(
            0, action_expert_config["number_high_level_command"], (batch_size, gpt_config["nb_timesteps"])
        ),
        "diffusion_step": torch.rand(batch_size, gpt_config["nb_timesteps"]),
    }

    outputs = joint_model(attn_mask, inputs_all)

    print("Output shape:", outputs["actions"].shape, outputs["actions_embeds"].shape)
