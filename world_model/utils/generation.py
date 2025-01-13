"""
Autoregressive Image Generation System

This module implements autoregressive generation of image sequences based on:
1. Burnin context: Initial frames and actions that provide context
2. Future actions: Sequence of actions that guide future frame generation
3. Rolling context: A sliding window of tokens that maintains bounded memory usage

The generation process works by:
1. Starting with burnin frames/actions as initial context
2. Generating each future frame token by token
3. Using a rolling window to maintain bounded context size
4. Incorporating future actions between generated frames

Key Components:
- KV Cache: Optional caching mechanism for transformer key/value pairs
- Context Rolling: Sliding window mechanism for bounded memory
- Position Encodings: Spatial and temporal position information
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from tqdm.auto import tqdm


class Sampler:
    """Base class for sampling strategies."""

    def sample(self, probabilities: Tensor) -> Tensor:
        """Samples from the probabilities according to a specific strategy.

        Args:
            probabilities: A tensor of probabilities from which to sample.

        Returns:
            A tensor representing the sampled indices.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self) -> str:
        raise NotImplementedError(
            "Every sub-class of `Sampler` needs a string representation. " "For example, str(TopKSampler(k=5)) == 'top_5'"
        )


class ArgmaxSampler(Sampler):
    """Implements deterministic argmax sampling strategy.

    This sampler always selects the token with the highest probability.
    Useful for deterministic generation or testing purposes.
    """

    def __repr__(self) -> str:
        return "argmax"

    def sample(self, next_token_probabilities: Tensor) -> Tensor:
        """Samples by selecting the token with the highest probability.

        Args:
            next_token_probabilities: A tensor of already normalized probabilities.

        Returns:
            A tensor representing the indices of the sampled elements.
        """
        # Get the index of the highest probability token
        next_token = torch.argmax(next_token_probabilities, dim=-1, keepdim=True)
        return next_token


class TopKSampler(Sampler):
    """Implements Top-K sampling strategy on already normalized probabilities.

    Args:
            k: The number of highest probability tokens to consider for sampling.
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def __repr__(self) -> str:
        return f"top_{self.k}"

    def sample(self, next_token_probabilities: Tensor) -> Tensor:
        """Samples from the top K probabilities.

        Args:
            next_token_probabilities: A tensor of already normalized probabilities from which to sample.

        Returns:
            A tensor representing the indices of the sampled elements.
        """

        topk_probs, topk_tokens = torch.topk(next_token_probabilities, k=self.k, dim=-1, sorted=False)

        topk_renormalized_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        # Sample from the top k normalized probabilities
        next_token_idx = torch.multinomial(topk_renormalized_probs, num_samples=1)
        next_token = torch.gather(topk_tokens, -1, next_token_idx)

        return next_token


class KVCache(nn.Module):
    """
    Adapted from
    https://github.com/karpathy/nano-llama31/blob/06461cada7744a7da86a408f094549800b6bee3f/llama31.py#L141
    """

    def __init__(
        self, batch_size: int, seq_length: int, n_kv_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, n_kv_heads, seq_length, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos: int, xk: Tensor, xv: Tensor) -> Tuple[Tensor, Tensor]:
        if start_pos == -1:
            return xk, xv
        # changed from original implementation because shape in mup_GPT2 is (b nb_heads seq dim_head)
        seqlen = xk.size(2)
        self.cache_k[:, :, start_pos : start_pos + seqlen] = xk
        self.cache_v[:, :, start_pos : start_pos + seqlen] = xv
        xk = self.cache_k[:, :, : start_pos + seqlen]
        xv = self.cache_v[:, :, : start_pos + seqlen]
        return xk, xv


class ContextState(NamedTuple):
    """
    Holds the current state of the rolling context.

    Attributes:
        tokens: Token sequence combining visual and action tokens [B, T]
        spatial_pos: Spatial position encodings for tokens [B, T]
        temporal_pos: Temporal position encodings for tokens [B, T]
    """

    tokens: Tensor
    spatial_pos: Tensor
    temporal_pos: Tensor


class GenerationDims(NamedTuple):
    """
    Holds the key dimensions for generation.

    Attributes:
        batch_size: Number of sequences to generate in parallel
        height: Height of each frame in tokens
        width: Width of each frame in tokens
        n_visual_tokens: Total number of visual tokens per frame (height * width)
        n_action_tokens: Number of action tokens per timestep
        n_future_frames: Number of frames to generate
        max_context_size: Maximum number of tokens in rolling context
    """

    batch_size: int
    height: int
    width: int
    n_visual_tokens: int
    n_action_tokens: int
    n_future_frames: int
    max_context_size: int


@dataclass
class GenerationConfig:
    """Configuration for autoregressive image generation."""

    temperature: float = 1.0
    return_logits: bool = False
    use_kv_cache: bool = True
    verbose: bool = False


class ImageGenerator:
    """Handles autoregressive generation of image sequences."""

    def __init__(
        self, network: nn.Module, sampler: Sampler, sequence_adapter: nn.Module, max_rolling_context_frames: int
    ) -> None:
        self.network = network
        self.sampler = sampler
        self.sequence_adapter = sequence_adapter
        self.max_context_frames = max_rolling_context_frames

    def _extract_dimensions(self, visual_tokens: Tensor, action_tokens: Tensor) -> GenerationDims:
        """
        Extracts all relevant dimensions for the generation process.

        The context size is determined by:
        max_context_size = max_frames * (tokens_per_frame + action_tokens)

        This ensures we can store up to max_frames worth of visual and action tokens
        in our rolling context window.
        """
        batch_size, _, height, width = visual_tokens.shape
        n_future_actions = action_tokens.shape[1]
        n_action_tokens = action_tokens.shape[-1]
        n_visual_tokens = height * width

        max_context_size = self.max_context_frames * (n_visual_tokens + n_action_tokens)
        n_future_frames = n_future_actions + 1  # +1 for initial frame

        return GenerationDims(
            batch_size=batch_size,
            height=height,
            width=width,
            n_visual_tokens=n_visual_tokens,
            n_action_tokens=n_action_tokens,
            n_future_frames=n_future_frames,
            max_context_size=max_context_size,
        )

    def _setup_kv_cache(self, dims: GenerationDims) -> None:
        """
        Sets up key-value caching for transformer attention layers.

        KV Cache:
        - Stores the Key and Value projections for each token
        - Avoids recomputing these for previous tokens during generation
        - Cache size matches max_context_size to align with rolling context
        - Must be updated (rolled) when context is rolled
        """
        for block in self.network.transformer.h:
            layer = block.attn
            cache = KVCache(
                batch_size=dims.batch_size,
                seq_length=dims.max_context_size,
                n_kv_heads=layer.nb_heads,
                head_dim=layer.dim_model // layer.nb_heads,
                dtype=layer.c_attn.weight.dtype,
                device=layer.c_attn.weight.device,
            )
            layer.cache = cache

    def _prepare_initial_context(
        self,
        dims: GenerationDims,
        burnin_visual: Tensor,
        burnin_actions: Tensor,
    ) -> Tuple[ContextState, Dict]:
        """Prepares initial context by interleaving burnin tokens and computing positions."""
        # Interleave visual and action tokens from burnin sequence
        sequence_data = self.sequence_adapter(burnin_visual, burnin_actions)
        context = sequence_data["token_sequence"]

        # Compute temporal and spatial position indices
        position_data = self.sequence_adapter.compute_position_indices(
            dims.batch_size, dims.height, dims.width, dims.n_action_tokens, self.max_context_frames
        )

        spatial_pos = position_data["spatial_positions"]
        temporal_pos = position_data["temporal_positions"]

        # Trim to max_context_size if needed
        if context.shape[1] > dims.max_context_size:
            context = context[:, -dims.max_context_size :]
            spatial_pos = spatial_pos[:, : dims.max_context_size]
            temporal_pos = temporal_pos[:, : dims.max_context_size]

        return ContextState(context, spatial_pos, temporal_pos), position_data, sequence_data

    def _get_model_output(self, state: ContextState, prev_pos: int, cur_pos: int, use_cache: bool) -> Tensor:
        """
        Gets model predictions using either full context or cached computations.

        Two Modes:
        1. With KV Cache:
           - Only process new tokens (prev_pos:cur_pos)
           - Use cached keys/values for previous tokens
        2. Without KV Cache:
           - Process entire context each time
           - Less efficient but simpler

        Position indices are crucial for relative attention calculations.
        """
        if use_cache:
            return self.network(
                state.tokens[:, prev_pos:cur_pos],
                state.spatial_pos[:, prev_pos:cur_pos],
                state.temporal_pos[:, prev_pos:cur_pos],
                inference=True,
                start_pos=prev_pos,
            )

        return self.network(
            state.tokens,
            state.spatial_pos[:, : state.tokens.shape[1]],
            state.temporal_pos[:, : state.tokens.shape[1]],
            inference=True,
            start_pos=-1,
        )

    def _sample_next_token(self, logits: Tensor, temperature: float) -> Tuple[Tensor, Tensor]:
        """Sample the next token from model logits."""
        token_logits = logits[:, -1, :] / temperature
        token_probs = F.softmax(token_logits, dim=-1)
        next_token = self.sampler.sample(token_probs)
        return next_token, token_logits

    def _update_context_state(
        self, state: ContextState, next_token: Tensor, dims: GenerationDims, use_cache: bool
    ) -> ContextState:
        """
        Updates rolling context with new token and handles context window sliding.

        We mostly need to roll because the model has been trained with a fixed context size.
        The rolling is useful when working in a sliding window fashion at context size limit.
        Special care have to be taken to the positional embedding. In particular, we have
        to be careful to the temporal one as it increases with each frame while having a
        fixed limited determined at train time.

        Rolling Context Logic:
        1. Add new token to end of context
        2. If context exceeds max_context_size:
           - Remove oldest token, concat new to the end (left shift)
           - Roll position encodings to match
           - Roll KV cache

        The crucial temporal position update (temporal_pos[:, -1] += self.max_context_frames)
        maintains consistent temporal relationships in the rolling context. Here's why:

        Example with:
        - 2 patches per frame
        - 4 learned temporal embeddings (0,1,2,3)
        - max_context_frames = 3   (model max - 1)
        - max_context_size = 3 * 2 = 6

        Evolution of context:
        1. Initial frame:
            Tokens:        [P1 P2]        (P1,P2 are patches)
            Temporal Pos:  [0  0 ]         (same frame = same temporal pos)

        2. After second frame:
            Tokens:        [P1 P2 | P3 P4]    (| separates frames)
            Temporal Pos:  [0  0  | 1  1 ]

        3. After third frame:
            Tokens:        [P1 P2 | P3 P4 | P5 P6]
            Temporal Pos:  [0  0  | 1  1  | 2  2 ]

        4. Predicting the next patch of the fourth frame:
            Without increment:
                Tokens:        [P2 | P3 P4 | P5 P6 | P7]    (P1 dropped)
                Temporal Pos:  [0  | 1  1  | 2  2  | 0 ]    <- WRONG! New frame appears "earlier"

            With increment (+=max_context_frames):
                Tokens:        [P2 | P3 P4 | P5 P6 | P7]
                Temporal Pos:  [0  | 1  1  | 2  2  | 3 ]    <- CORRECT! Maintains ordering

        5. Continuing
            Tokens:        [P3 P4 | P5 P6 | P7 P8]
            Temporal Pos:  [1  1  | 2  2  | 3  3 ]

        At this point we genererated the entire frame and before begining the cycle again from step 3
        for next frame we need to reset the temporal:
            Temporal Pos:  [0  0  | 1  1  | 2  2 ]

        NOTE: Alternative approach
        Instead of rolling patch by patch, we could:
            1. Use the entire model context size (e.g., 4 frames * num patches per frame)
            2. Generate all patches for frames 1-4
            3. Once frame 4 is complete, drop frame 1 entirely
            4. reset pos embedding
            4. cycle to 1.
        This would be simpler
        """
        tokens = torch.cat([state.tokens, next_token], dim=1)
        spatial_pos = state.spatial_pos
        temporal_pos = state.temporal_pos

        if tokens.shape[1] > dims.max_context_size:

            tokens = tokens[:, 1:]
            spatial_pos = spatial_pos.roll(-1, dims=1)
            temporal_pos = temporal_pos.roll(-1, dims=1)
            temporal_pos[:, -1] += self.max_context_frames

            raise NotImplementedError("rolling kv caching not implemented")

            if use_cache:
                self._roll_kv_cache()

        return ContextState(tokens, spatial_pos, temporal_pos)

    def _process_generated_frame(
        self, context: Tensor, dims: GenerationDims, frame_logits: Optional[List[Tensor]] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Extract and process the latest generated frame."""
        frame = context[:, -dims.n_visual_tokens :]
        frame = rearrange(frame, "b (h w) -> b h w", h=dims.height)

        if not frame_logits:
            return frame, None

        logits = torch.stack(frame_logits, dim=1)
        logits = rearrange(logits, "b (h w) v -> b h w v", h=dims.height)
        return frame, logits

    def _add_action_tokens(
        self, state: ContextState, future_actions: Tensor, frame_idx: int, dims: GenerationDims, use_cache: bool
    ) -> ContextState:
        """
        Adds conditioning action tokens between generated frames.

        Action Token Flow:
        1. Get actions for next frame
        2. Shift action token indices to get their actual codebook index
        3. Add to context
        4. Roll context if needed
        """
        next_actions = future_actions[:, frame_idx]
        next_actions = self.sequence_adapter.action_shifts + next_actions
        tokens = torch.cat([state.tokens, next_actions], dim=1)

        if tokens.shape[1] > dims.max_context_size:
            offset = tokens.size(1) - dims.max_context_size
            tokens = tokens[:, offset:]

            if use_cache:
                self._roll_kv_cache(offset)

        return ContextState(tokens, state.spatial_pos, state.temporal_pos)

    def _roll_kv_cache(self, offset: int = 1) -> None:
        """Roll the Key-Value cache for all transformer blocks."""
        for block in self.network.transformer.h:
            block.attn.cache.cache_k = block.attn.cache.cache_k.roll(-offset, dims=2)
            block.attn.cache.cache_v = block.attn.cache.cache_v.roll(-offset, dims=2)

    @torch.no_grad()
    def generate(
        self,
        burnin_visual_tokens: Tensor,
        burnin_action_tokens: Tensor,
        future_action_tokens: Tensor,
        config: Optional[GenerationConfig] = None,
    ) -> Dict[str, Tensor]:
        """
        Generate image sequences autoregressively.

        Args:
            burnin_visual_tokens: Visual tokens for context, shape [B, T_burnin, H, W]
            burnin_action_tokens: Action tokens for context, shape [B, T_burnin, K]
            future_action_tokens: Future action tokens, shape [B, T_future, K]
            config: Generation configuration parameters

        Returns:
            Dictionary containing generated visual tokens and optional logits
        """
        config = config or GenerationConfig()
        dims = self._extract_dimensions(burnin_visual_tokens, future_action_tokens)

        if config.use_kv_cache:
            self._setup_kv_cache(dims)

        try:
            # Initialize generation
            state, position_data, sequence_data = self._prepare_initial_context(
                dims, burnin_visual_tokens, burnin_action_tokens
            )

            generated_frames = []
            generated_logits = []
            prev_pos = 0
            cur_pos = state.tokens.shape[1]

            # Main generation loop
            frame_range = tqdm(range(dims.n_future_frames), disable=not config.verbose)
            for frame_idx in frame_range:
                frame_logits = [] if config.return_logits else None

                # TODO: change of frame, resetting position indices
                # rolling_spatial_positions = intial_spatial_positions
                # rolling_temporal_positions = initial_temporal_positions

                # Generate tokens for current frame
                for _ in range(dims.n_visual_tokens):
                    # Get next token
                    logits = self._get_model_output(state, prev_pos, cur_pos, config.use_kv_cache)
                    next_token, token_logits = self._sample_next_token(logits, config.temperature)

                    if config.return_logits:
                        frame_logits.append(token_logits)

                    # Update state
                    if config.use_kv_cache:
                        prev_pos = min(cur_pos, dims.max_context_size - 1)
                        cur_pos = min(cur_pos + 1, dims.max_context_size)

                    state = self._update_context_state(state, next_token, dims, config.use_kv_cache)

                # Process generated frame
                frame, frame_logits_tensor = self._process_generated_frame(state.tokens, dims, frame_logits)
                generated_frames.append(frame)
                if frame_logits_tensor is not None:
                    generated_logits.append(frame_logits_tensor)

                # Add action tokens for next frame if not last frame
                if frame_idx + 1 < dims.n_future_frames:
                    state = self._add_action_tokens(state, future_action_tokens, frame_idx, dims, config.use_kv_cache)
                    if config.use_kv_cache:
                        cur_pos += dims.n_action_tokens

        finally:
            if config.use_kv_cache:
                for block in self.network.transformer.h:
                    block.attn.cache = None

        # Prepare output
        result = {
            "visual_tokens": torch.stack(generated_frames, dim=1),
            "sequence_data": sequence_data,
            "position_data": position_data,
        }

        if generated_logits:
            result["visual_logits"] = torch.stack(generated_logits, dim=1)

        return result
