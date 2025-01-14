import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import FloatTensor, LongTensor

from world_model.gpt2.joint_model import JointModel
from world_model.gpt2.mup_dit import ActionEncoder


class Vai0rbis(nn.Module):

    def __init__(
        self,
        action_config: OmegaConf,
        joint_config: OmegaConf,
        finetuning_timesteps: int = 8,
        flow_sig_min: float = 0.001,
    ) -> None:
        super().__init__()
        # My parameters
        self.flow_sig_min = flow_sig_min

        # Load the models
        self.action_encoder: ActionEncoder = instantiate(action_config)
        self.joint_model: JointModel = instantiate(joint_config)

        # Video generation parameters
        self.nb_tokens_per_timestep = self.joint_model.gpt.nb_tokens_per_timestep
        self.context_length = finetuning_timesteps

        # Action parameters
        self.action_dim = self.action_encoder.action_dim
        self.action_horizon = self.action_encoder.action_horizon
        self.action_hidden_dim = self.action_encoder.action_hidden_dim

        # TODO: should this be setup with mup?
        self.action_decoder = nn.Linear(self.action_hidden_dim, self.action_dim)

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

        # inference with noisy action
        # [Batch_Size, Embed_Dim]
        action_embeds = self.action_encoder(actions=psi_t, high_level_command=high_level_command, diffusion_step=t)

        action_embeds = self.joint_model(
            attention_mask=self.attn_mask,
            inputs_all={
                "visual_tokens": visual_tokens,
                "action_embeds": action_embeds,
            },
        )["action"]

        # [Batch_Size, Horizon_Steps, Action_Dim]
        v_psi = self.action_decoder(action_embeds)

        # compare to true velocity
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)
