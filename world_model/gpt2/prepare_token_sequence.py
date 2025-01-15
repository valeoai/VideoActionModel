from typing import Dict

import torch
from einops import rearrange, repeat
from torch import Tensor


def compute_position_indices(batch_size: int, num_frames: int, height: int, width: int) -> Dict[str, Tensor]:
    """
    Compute spatial positions, temporal positions, for a given batch size, frame size, and number of frames.

    This static method can be used independently during inference for autoregressive frame generation.

    Args:
        batch_size: Number of samples in the batch.
        height: Height of each frame.
        width: Width of each frame.
        num_frames: Number of frames to generate indices for.

    Returns:
        A dictionary containing:
            - spatial_positions: Tensor of spatial positions for each token, shape [B, T*H*W].
            - temporal_positions: Tensor of temporal positions for each token, shape [B, T*H*W].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_visual_tokens = height * width
    total_tokens_per_frame = num_visual_tokens

    spatial_positions = torch.arange(height * width, device=device)
    spatial_positions = repeat(spatial_positions, "s -> b (t s)", b=batch_size, t=num_frames)

    temporal_positions = torch.arange(num_frames, device=device)
    temporal_positions = repeat(temporal_positions, "t -> b (t s)", b=batch_size, s=total_tokens_per_frame)

    return {"spatial_positions": spatial_positions, "temporal_positions": temporal_positions}


def prepare_token_sequence(visual_tokens: Tensor) -> Dict[str, torch.Tensor]:

    position_indices = compute_position_indices(*visual_tokens.shape)

    visual_tokens = rearrange(visual_tokens, "b t h w -> b (t h w)")

    return {"token_sequence": visual_tokens, **position_indices}


def prepare_AR_token_sequences(visual_tokens: Tensor) -> Dict[str, torch.Tensor]:

    sequence_data = prepare_token_sequence(visual_tokens)

    # Create input_tokens by taking all but the last token (shifting by one)
    input_data = {
        "token_sequence": sequence_data["token_sequence"][:, :-1],
        "spatial_positions": sequence_data["spatial_positions"][:, :-1],
        "temporal_positions": sequence_data["temporal_positions"][:, :-1],
    }

    # Create target_tokens by taking all but the first token (shifting by one)
    target_data = {
        "token_sequence": sequence_data["token_sequence"][:, 1:],
    }

    return input_data, target_data
