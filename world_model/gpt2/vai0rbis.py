import os
from collections import OrderedDict
from typing import Dict, Tuple

import mup
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import BoolTensor, LongTensor, Tensor
from tqdm import tqdm

from world_model.gpt2.joint_model import JointModel
from world_model.gpt2.mup_action_expert import MupActionExpert
from world_model.gpt2.mup_gpt2 import MupGPT2

mupShapes = str | Dict[str, Tuple[int, ...]]


class Vai0rbis(nn.Module):
    """
    This module coordinates the training of a JointModel.

    Notable it implements:
    - the flow matching loss for action prediction.
    - The denoise diffusion process for action prediction.
    """

    def __init__(
        self,
        gpt_config: OmegaConf,
        gpt_mup_base_shapes: mupShapes | None,
        gpt_checkpoint_path: str | None,
        action_config: OmegaConf,
        action_mup_base_shapes: mupShapes | None,
        action_checkpoint_path: str | None,
        finetuning_timesteps: int = 8,
        num_inference_steps: int = 10,
        flow_sig_min: float = 0.001,
        final_action_clip_value: float | None = None,
        action_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        # My parameters
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.final_action_clip_value = final_action_clip_value
        self.context_length = finetuning_timesteps
        self.action_scaling = action_scaling

        # Load the models
        ## Video generation model
        self.gpt: MupGPT2 = instantiate(gpt_config)
        self.gpt_mup_base_shapes = gpt_mup_base_shapes
        if gpt_checkpoint_path is not None:
            gpt_state_dict = self._load_ckpt(gpt_checkpoint_path, key="network.")
            self.gpt.load_state_dict(gpt_state_dict)
            mup.set_base_shapes(self.gpt, gpt_mup_base_shapes, rescale_params=False)
            self.gpt.requires_grad_(False)
        else:
            mup.set_base_shapes(self.gpt, gpt_mup_base_shapes)
            self.gpt.apply(self.gpt._init_weights)  # re-initialize after set_base_shapes
        ## Action model
        self.action_expert: MupActionExpert = instantiate(action_config)
        self.action_mup_base_shapes = action_mup_base_shapes
        if action_checkpoint_path is not None:
            action_state_dict = self._load_ckpt(action_checkpoint_path)
            self.action_expert.load_state_dict(action_state_dict)
            mup.set_base_shapes(self.action_expert, action_mup_base_shapes, rescale_params=False)
            self.action_expert.requires_grad_(False)
        else:
            mup.set_base_shapes(self.action_expert, action_mup_base_shapes)
            self.action_expert.apply(self.action_expert._init_weights)  # re-initialize after set_base_shapes
        ## Joint model
        self.joint_model = JointModel(self.gpt, self.action_expert)

        # Video generation parameters
        self.nb_tokens_per_timestep = self.gpt.nb_tokens_per_timestep

        # Action parameters
        self.action_dim = self.action_expert.action_dim
        self.action_horizon = self.action_expert.action_horizon
        self.action_hidden_dim = self.action_expert.embedding_dim

        self.build_attention_mask()

    def _load_ckpt(self, ckpt: str, key: str | None) -> OrderedDict:
        ckpt = torch.load(os.path.expanduser(os.path.expandvars(ckpt)), map_location="cpu")
        if key is not None:
            # We need to remove the prefix "network." from the keys of the state_dict
            state_dict = OrderedDict()
            for k, v in ckpt["state_dict"].items():
                state_dict[k.replace(key, "")] = v
        else:
            state_dict = ckpt["state_dict"]
        return state_dict

    def build_attention_mask(self) -> None:
        """
        Builds the attention mask for the joint model.

        - Causal attention mask for the visual tokens.
        - Bi-directional attention mask for the action tokens withing the action horizon.
        - Full-attention between the action tokens and the visual tokens of the context.

        Attention mask (this same for each sample of the batch, so we ignore the batch dimension for visualazation).

        For a max context length of 3 and 3 tokens per image and an action horizon of 2, the mask looks like this:

        V11 V12 V13 V21 V22 V23 V31 V32 V33 H11 H12 H21 H22 H31 H32
         x
         x   x
         x   x   x
         x   x   x   x
         x   x   x   x   x
         x   x   x   x   x   x
         x   x   x   x   x   x   x
         x   x   x   x   x   x   x   x
         x   x   x   x   x   x   x   x   x

         x   x   x                          x    x
         x   x   x                          x    x
         x   x   x   x   x   x                       x    x
         x   x   x   x   x   x                       x    x
         x   x   x   x   x   x   x   x                       x    x
         x   x   x   x   x   x   x   x                       x    x
        """
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

    def build_inference_attention_mask(self, context_length: int, device: torch.device | str) -> BoolTensor:
        """
        Builds the attention mask for the joint model during inference.

        - Causal attention mask for the visual tokens.
        - Full attention for the actions tokens
        """
        visual_seqlen = self.nb_tokens_per_timestep * context_length
        action_seq_len = self.action_horizon  # this time we predict only one step
        seqlen = visual_seqlen + action_seq_len
        attn_mask = torch.zeros(seqlen, seqlen, dtype=torch.bool, device=device)

        # Causal mask for the visual tokens
        for i in range(visual_seqlen):
            attn_mask[i, : i + 1] = True

        # Bi-directional mask for the action tokens + action tokens can attend to all visual tokens
        attn_mask[visual_seqlen:] = True

        return attn_mask

    def psi_t(
        self,
        x: Tensor,
        x1: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Conditional Flow"""
        t = t[:, :, None, None]  # (B, finetuning_timesteps, 1, 1)
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        visual_tokens: LongTensor,
        high_level_command: LongTensor,
        actions: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Flow matching loss for action prediction"""
        # noisy action
        # [Batch_Size, finetuning_timesteps, Horizon_Steps, Action_Dim]
        x0 = torch.randn_like(actions, device=actions.device)
        x1 = actions / self.action_scaling
        psi_t = self.psi_t(x0, x1, t).type_as(x1)

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
        return torch.mean((v_psi - d_psi) ** 2).type_as(x1)

    def forward_inference(
        self,
        visual_tokens: LongTensor,
        high_level_command: LongTensor,
        dtype: torch.dtype,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Inference for action prediction.

        We start from a random action and integrate the dynamics using a forward Euler scheme.
        """
        device = visual_tokens.device
        bsz, context_length, *_ = visual_tokens.size()
        assert context_length <= self.context_length

        # sample pure action noise
        action = torch.randn((bsz, 1, self.action_horizon, self.action_dim), device=device, dtype=dtype)

        # attn_mask for inference
        attn_mask = self.build_inference_attention_mask(context_length, device=device)

        # Init KV cache
        self.joint_model.init_kv_cache()

        # forward euler integration ---
        delta_t = 1.0 / self.num_inference_steps
        t = torch.zeros((bsz, 1), device=device, dtype=dtype)
        for _ in tqdm(range(self.num_inference_steps), "Euler integration", disable=not verbose):
            action_vel = self.joint_model(
                attention_mask=attn_mask,
                inputs_all={
                    "visual_tokens": visual_tokens,
                    "noisy_actions": action,
                    "high_level_command": high_level_command,
                    "diffusion_step": t,
                },
            )["actions"]
            # decode action: [Batch_Size, Horizon_Steps, Action_Dim]
            action += delta_t * action_vel
            t += delta_t

        # cleanup KV cache
        self.joint_model.cleanup_kv_cache()

        # clamp final output if specified
        if self.final_action_clip_value is not None:
            action = torch.clamp(
                action,
                -self.final_action_clip_value,
                self.final_action_clip_value,
            )
        return action * self.action_scaling


class Vai0rbisInference(Vai0rbis):
    """Helper class to perform inference with the Vai0rbis model."""

    def forward(
        self,
        visual_tokens: LongTensor,
        high_level_command: LongTensor,
        dtype: torch.dtype,
        verbose: bool = False,
    ) -> torch.Tensor:
        return super().forward_inference(visual_tokens, high_level_command, dtype, verbose)


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
    action_expert_config = {
        "_target_": "world_model.gpt2.mup_action_expert.MupActionExpert",
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

    visual_tokens = torch.randint(0, gpt_config.vocabulary_size, (batch_size, gpt_config.nb_timesteps, height, width))
    high_level_command = torch.randint(
        0, action_expert_config.number_high_level_command, (batch_size, gpt_config.nb_timesteps)
    )
    actions = torch.randn(
        batch_size, gpt_config.nb_timesteps, action_expert_config.action_horizon, action_expert_config.action_dim
    )
    t = torch.rand(batch_size, gpt_config.nb_timesteps)

    loss = vai0rbis(visual_tokens, high_level_command, actions, t)
    print("Loss:", loss)

    visual_tokens = torch.randint(0, gpt_config.vocabulary_size, (batch_size, 5, height, width))
    high_level_command = torch.randint(0, action_expert_config.number_high_level_command, (batch_size, 1))
    actions_inference = vai0rbis.forward_inference(visual_tokens, high_level_command, torch.float32, verbose=True)
