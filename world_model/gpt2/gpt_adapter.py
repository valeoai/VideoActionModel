from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat


class GPTAdapter(nn.Module):
    """
    This adapter handles the combination of visual and action tokens by interleaving them into a single sequence
    [v0,a0,v1,a1,...vn,an] suitable for GPT-like models.

    Since visual and action tokens originate from separate vocabularies, there's a potential overlap in their
    numerical values. To prevent this, action tokens are shifted by adding the size of the visual vocabulary,
    effectively creating a unified vocabulary space. This shifting ensures that each token, whether visual or
    action, has a unique representation.
    """

    def __init__(self, visual_vocab_size: int) -> None:
        """
        Initializes the GPTAdapter with specified visual and action vocabulary sizes.

        Args:
            visual_vocab_size: Size of the visual vocabulary.
        """
        super().__init__()
        self.visual_vocab_size = visual_vocab_size

    @staticmethod
    def compute_position_indices(batch_size: int, height: int, width: int, num_frames: int) -> Dict[str, torch.Tensor]:
        """
        Compute spatial positions, temporal positions,
        and visual tokens mask for a given batch size, frame size, and number of frames.

        This static method can be used independently during inference for autoregressive frame generation.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of each frame.
            width: Width of each frame.
            num_frames: Number of frames to generate indices for.

        Returns:
            A dictionary containing:
                - spatial_positions: Tensor of spatial positions for each token, shape [B, T * (H*W + K)].
                - temporal_positions: Tensor of temporal positions for each token, shape [B, T * (H*W + K)].
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_visual_tokens = height * width

        spatial_positions = torch.arange(height * width, device=device)
        spatial_positions = repeat(spatial_positions, "s -> b (t s)", b=batch_size, t=num_frames)

        temporal_positions = torch.arange(num_frames, device=device)
        temporal_positions = repeat(temporal_positions, "t -> b (t s)", b=batch_size, s=num_visual_tokens)

        return {
            "spatial_positions": spatial_positions,
            "temporal_positions": temporal_positions,
        }

    def forward(self, visual_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_tokens: Tensor of visual tokens with shape [B, T, H, W].

        Returns:
            A dictionary containing:
                - token_sequence: Interleaved sequence of visual and action tokens.
                - spatial_positions: Spatial position encodings for each token.
                - temporal_positions: Temporal position encodings for each token.
        """
        B, T, H, W = visual_tokens.shape

        visual_tokens = rearrange(visual_tokens, "b t h w -> b (t h w)")

        position_indices = self.compute_position_indices(B, H, W, T)

        return {"token_sequence": visual_tokens, **position_indices}
