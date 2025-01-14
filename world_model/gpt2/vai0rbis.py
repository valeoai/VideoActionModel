from typing import Dict, Tuple

import mup
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import FloatTensor, LongTensor

from world_model.gpt2.joint_model import JointModel
from world_model.gpt2.mup_action_expert import MupActionExpert
from world_model.gpt2.mup_gpt2 import MupGPT2

mupShapes = Dict[str, Tuple[int, ...]]


class Vai0rbis(nn.Module):

    def __init__(
        self,
        gpt_config: OmegaConf,
        gpt_mup_base_shapes: mupShapes | None,
        action_config: OmegaConf,
        action_mup_base_shapes: mupShapes | None,
        finetuning_timesteps: int = 8,
        flow_sig_min: float = 0.001,
    ) -> None:
        super().__init__()
        # My parameters
        self.flow_sig_min = flow_sig_min

        # Load the models
        ## Video generation model
        self.gpt: MupGPT2 = instantiate(gpt_config)
        mup.set_base_shapes(self.gpt, gpt_mup_base_shapes)
        self.gpt.apply(self.gpt._init_weights)  # re-initialize after set_base_shapes
        ## Action model
        self.action_expert: MupActionExpert = instantiate(action_config)
        mup.set_base_shapes(self.action_expert, action_mup_base_shapes)
        self.action_expert.apply(self.action_expert._init_weights)  # re-initialize after set_base_shapes
        ## Joint model
        self.joint_model = JointModel(self.gpt, self.action_expert)

        # Video generation parameters
        self.nb_tokens_per_timestep = self.gpt.nb_tokens_per_timestep
        self.context_length = finetuning_timesteps

        # Action parameters
        self.action_dim = self.action_expert.action_dim
        self.action_horizon = self.action_expert.action_horizon
        self.action_hidden_dim = self.action_expert.embedding_dim

        self.build_attention_mask()

    def build_attention_mask(
        self,
    ) -> None:
        context_length = self.context_length
        action_horizon = self.action_horizon
        nb_tokens_per_timestep = self.nb_tokens_per_timestep

        # Number of visual tokens in the entire sequence
        visual_seqlen = self.nb_tokens_per_timestep * self.context_length
        # Number of action tokens in the entire sequence
        action_seq_len = self.action_horizon * self.context_length
        # Total number of tokens in the sequence
        seqlen = visual_seqlen + action_seq_len
        attn_mask = torch.zeros(seqlen, seqlen, dtype=torch.bool)

        # Causal mask for the visual tokens
        for i in range(visual_seqlen):
            attn_mask[i, : i + 1] = True

        # Bi-directional mask for each set of action tokens
        for i in range(context_length):
            attn_mask[
                visual_seqlen + i * action_horizon : visual_seqlen + (i + 1) * action_horizon,
                visual_seqlen + i * action_horizon : visual_seqlen + (i + 1) * action_horizon,
            ] = True

        # Actions tokens at timestep t can attend to visual tokens at timestep <= t
        for i in range(context_length):
            attn_mask[
                visual_seqlen + i * action_horizon : visual_seqlen + (i + 1) * action_horizon,
                : nb_tokens_per_timestep * (i + 1),
            ] = True

        self.register_buffer("attn_mask", attn_mask)

    def psi_t(
        self,
        x: FloatTensor,
        x1: FloatTensor,
        t: FloatTensor,
    ) -> FloatTensor:
        """Conditional Flow"""
        t = t[:, None, None]  # (B, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        visual_tokens: LongTensor,
        high_level_command: LongTensor,
        actions: FloatTensor,
        t: FloatTensor,
    ) -> FloatTensor:
        """flow matching loss for action prediction, no use of kv cache"""
        # noisy action
        # [Batch_Size, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=t.device, dtype=t.dtype)
        x1 = actions
        psi_t = self.psi_t(x0, x1, t)

        v_psi = self.joint_model(
            attention_mask=self.attn_mask,
            inputs_all={
                "visual_tokens": visual_tokens,
                "noisy_actions": psi_t,
                "high_level_command": high_level_command,
                "diffusion_step": t,
            },
        )["actions"]

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)


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
    gpt_config = OmegaConf.create(gpt_config)
    action_expert_config = OmegaConf.create(action_expert_config)

    vai0rbis = Vai0rbis(
        gpt_config=gpt_config,
        gpt_mup_base_shapes=None,
        action_config=action_expert_config,
        action_mup_base_shapes=None,
        finetuning_timesteps=gpt_config.nb_timesteps,
    )

    batch_size = 3

    visual_tokens = torch.randint(
        0, gpt_config.vocabulary_size, (batch_size, gpt_config.nb_timesteps, height, width)
    )
    high_level_command = torch.randint(0, action_expert_config.number_high_level_command, (batch_size,))
    actions = torch.randn(
        batch_size, gpt_config.nb_timesteps, action_expert_config.action_horizon, action_expert_config.action_dim
    )
    t = torch.rand(batch_size)

    loss = vai0rbis(visual_tokens, high_level_command, actions, t)
