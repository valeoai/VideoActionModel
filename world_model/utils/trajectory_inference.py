import yaml
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from hydra.utils import instantiate
import mup

from world_model.utils.generation import ArgmaxSampler


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    """util to load statedict"""
    result = dict()
    for k, v in state_dict.items():
        tokens = k.split('.')
        if tokens[0] == prefix:
            tokens = tokens[1:]
            key = '.'.join(tokens)
            result[key] = v
    return result


def load_world_model(checkpoint_path, model_config, device, inference_sha=None):
    checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))

    network = instantiate(model_config.network)
    mup.set_base_shapes(network, base=model_config.mup_base_shapes)
    sequence_adapter = instantiate(model_config.sequence_adapter)

    state_dict = remove_prefix(checkpoint_data['state_dict'], 'network')
    network.load_state_dict(state_dict, strict=True)

    return network, sequence_adapter


def load_trajectory_model(
    ckpt_file_path,
    config_file_path,
    mup_base_shapes_path,
    device='cuda',
):
    with open(config_file_path, 'r') as file:
        training_logged_config = yaml.safe_load(file)
    training_logged_config = OmegaConf.create(training_logged_config)

    model_config = training_logged_config.model

    model_config.mup_base_shapes = mup_base_shapes_path

    network, sequence_adapter = load_world_model(
        ckpt_file_path,
        model_config,
        device=device
    )
    network = network.to(device)
    sequence_adapter = sequence_adapter.to(device)

    network.eval()
    sequence_adapter.eval()

    network.requires_grad_(False)
    sequence_adapter.requires_grad_(False)

    return network, sequence_adapter


class WorldModelTrajectoryInference(nn.Module):
    def __init__(self, network, sequence_adapter, sampler=None):
        super().__init__()

        self.network = network
        self.sequence_adapter = sequence_adapter
        self.sampler = sampler or ArgmaxSampler()

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the next token from model logits."""
        token_logits = logits[:, -1, :] / temperature
        token_probs = F.softmax(token_logits, dim=-1)
        next_token = self.sampler.sample(token_probs)
        return next_token

    def forward(self, visual_tokens, command, trajectory_tokens=None, temperature=1.):
        """
        Args:
            visual_tokens: B, T, H, W
            command: B, T
            trajectory_tokens: B, T-1, 2


        Output:
            predicted_trajectory_tokens: B, 2

        """

        b, t, h, w = visual_tokens.shape

        if command.shape[1] != t:
            raise ValueError("There should be as many commands as there are frames")

        visual_tokens = rearrange(visual_tokens, 'b t h w -> b t (h w)')

        if trajectory_tokens is not None:
            if t <= 1:
                raise ValueError("if trajectory_tokens != None, visual_tokens must be for at least 2 frames")
            if trajectory_tokens.shape[1] != t - 1:
                raise ValueError("visual_tokens: B, T, H, W  | trajectory_tokens: B, T-1, 2")
            trajectory_tokens_shifted = trajectory_tokens + self.sequence_adapter.action_shifts
            combined_tokens = torch.cat((visual_tokens[:, :-1, :], trajectory_tokens_shifted), dim=-1)  # B, T-1, S
            token_sequence = rearrange(combined_tokens, 'b t s -> b (t s)')  # B, (T-1) * S
            token_sequence = torch.cat([token_sequence, visual_tokens[:, -1, :]], dim=-1)  # B, (T-1) * S + H*W
        else:
            token_sequence = visual_tokens
            token_sequence = rearrange(token_sequence, 'b t s -> b (t s)')

        # generate spatial_positions, temporal_positions and visual_tokens_mask
        full_position_data = self.sequence_adapter.compute_position_indices(
            b, h, w, 2, t
        )

        action_codebook_start_idx = self.sequence_adapter.visual_vocab_size
        for i in range(2):
            # trimming full position data to our current len
            # the full_position_data corresponding to the entire len after the full prediction
            pos_data = {key: val[:, :-2 + i] for key, val in full_position_data.items()}

            logits = self.network(
                token_sequence,
                pos_data['spatial_positions'],
                pos_data['temporal_positions'],
                pos_data['visual_tokens_mask'],
                command,
                inference=True
            )

            vocab_size = self.sequence_adapter.action_vocab_sizes[i]
            action_codebook_end_idx = action_codebook_start_idx + vocab_size
            action_i_logits = logits[..., action_codebook_start_idx:action_codebook_end_idx]
            next_token = self._sample_next_token(
                action_i_logits,
                temperature
            )
            action_codebook_start_idx = action_codebook_end_idx

            token_sequence = torch.cat(
                [token_sequence, next_token],
                dim=-1
            )

        return token_sequence[:, -2:]
