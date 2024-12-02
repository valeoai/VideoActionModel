from typing import Dict, List

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

    Example:
        If the visual vocabulary size is 1000 and the action vocabulary sizes are [100, 200] for two different
        action types (e.g., speeding and curvature), the shifting would work as follows:

        - Visual tokens range from 0 to 999.
        - Action tokens from the first type would be shifted to start from 1000,
            making a first type action token '0' become '1000'.
        - Action tokens from the second type would be further shifted to start from 1100,
            making a second type action token '0' become '1100'.
        This ensures that all tokens have unique values and can be distinguished by the model.
    """

    def __init__(self, visual_vocab_size: int, action_vocab_sizes: List[int]) -> None:
        """
        Initializes the GPTAdapter with specified visual and action vocabulary sizes.

        Args:
            visual_vocab_size: Size of the visual vocabulary.
            action_vocab_sizes: List containing the sizes of each action token vocabulary.
        """
        super().__init__()
        self.visual_vocab_size = visual_vocab_size
        self.action_vocab_sizes = action_vocab_sizes

        action_shifts = torch.tensor(self._calculate_action_shifts(), dtype=torch.long)
        self.register_buffer("action_shifts", action_shifts)

    def _calculate_action_shifts(self) -> List[int]:
        """
        Calculates shifts needed for action token indices based on visual and action vocabularies' sizes.

        Returns:
            Cumulative shifts for each action token type.
        """
        shifts = [self.visual_vocab_size]
        total_shift = self.visual_vocab_size
        for size in self.action_vocab_sizes[:-1]:
            total_shift += size
            shifts.append(total_shift)
        return shifts

    @staticmethod
    def compute_position_indices(
        batch_size: int, height: int, width: int, num_action_tokens: int, num_frames: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute spatial positions, temporal positions,
        and visual tokens mask for a given batch size, frame size, and number of frames.

        This static method can be used independently during inference for autoregressive frame generation.

        Args:
            batch_size: Number of samples in the batch.
            height: Height of each frame.
            width: Width of each frame.
            num_action_tokens: Number of action tokens per frame.
            num_frames: Number of frames to generate indices for.

        Returns:
            A dictionary containing:
                - spatial_positions: Tensor of spatial positions for each token, shape [B, T * (H*W + K)].
                - temporal_positions: Tensor of temporal positions for each token, shape [B, T * (H*W + K)].
                - visual_tokens_mask: Boolean mask indicating which tokens are visual (True)
                  vs action (False), shape [B, T * (H*W + K)].
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_visual_tokens = height * width
        total_tokens_per_frame = num_visual_tokens + num_action_tokens

        spatial_position_visual = torch.arange(height * width, device=device)
        spatial_position_action = torch.arange(num_action_tokens, device=device) + num_visual_tokens
        spatial_position = torch.cat((spatial_position_visual, spatial_position_action), dim=0)
        spatial_positions = repeat(spatial_position, "s -> b (t s)", b=batch_size, t=num_frames)

        temporal_positions = torch.arange(num_frames, device=device)
        temporal_positions = repeat(temporal_positions, "t -> b (t s)", b=batch_size, s=total_tokens_per_frame)

        visual_tokens_mask = repeat(spatial_position < num_visual_tokens, "s -> b (t s)", b=batch_size, t=num_frames)

        return {
            "spatial_positions": spatial_positions,
            "temporal_positions": temporal_positions,
            "visual_tokens_mask": visual_tokens_mask,
        }

    def forward(self, visual_tokens: torch.Tensor, action_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Adapts visual and action tokens into a unified sequence suitable for GPT models.
        This method interleaves flattened visual tokens with shifted action tokens into a single sequence.

        Args:
            visual_tokens: Tensor of visual tokens with shape [B, T, H, W].
            action_tokens: Tensor of action tokens with shape [B, T, K].

        Returns:
            A dictionary containing:
                - token_sequence: Interleaved sequence of visual and action tokens.
                - spatial_positions: Spatial position encodings for each token.
                - temporal_positions: Temporal position encodings for each token.
                - visual_tokens_mask: Boolean mask indicating which tokens are visual (True) vs action (False).
        """
        B, T, H, W = visual_tokens.shape
        _, _, K = action_tokens.shape

        visual_tokens = rearrange(visual_tokens, "b t h w -> b t (h w)")
        action_tokens_shifted = action_tokens + self.action_shifts

        combined_tokens = torch.cat((visual_tokens, action_tokens_shifted), dim=-1)
        interleaved_token_sequence = rearrange(combined_tokens, "b t s -> b (t s)")

        position_indices = self.compute_position_indices(B, H, W, K, T)

        return {"token_sequence": interleaved_token_sequence, **position_indices}
