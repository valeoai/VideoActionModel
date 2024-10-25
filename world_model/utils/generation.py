import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Sampler:
    """Base class for sampling strategies."""

    def sample(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Samples from the probabilities according to a specific strategy.

        Args:
            probabilities: A tensor of probabilities from which to sample.

        Returns:
            A tensor representing the sampled indices.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __repr__(self):
        raise NotImplementedError(
            "Every sub-class of `Sampler` needs a string representation. "
            "For example, str(TopKSampler(k=5)) == 'top_5'"
        )


class TopKSampler(Sampler):
    """Implements Top-K sampling strategy on already normalized probabilities.

    Args:
            k: The number of highest probability tokens to consider for sampling.
    """

    def __init__(self, k: int):
        self.k = k

    def __repr__(self):
        return f"top_{self.k}"

    def sample(self, next_token_probabilities: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, batch_size, seq_length, n_kv_heads, head_dim, dtype, device):
        super().__init__()
        cache_shape = (batch_size, seq_length, n_kv_heads, head_dim)
        self.register_buffer("cache_k", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("cache_v", torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, start_pos, xk, xv):
        seqlen = xk.size(1)
        self.cache_k[:, start_pos: start_pos + seqlen] = xk
        self.cache_v[:, start_pos: start_pos + seqlen] = xv
        xk = self.cache_k[:, : start_pos + seqlen]
        xv = self.cache_v[:, : start_pos + seqlen]
        return xk, xv


@torch.no_grad()
def autoregressive_image_sequence_generation(
    network,
    sampler,
    sequence_adapter,
    burnin_visual_tokens,
    burnin_action_tokens,
    future_action_tokens,
    max_rolling_context_frames,
    temperature=1.0,
    return_logits=False,
    use_kv_cache=False,
    verbose=False,
):
    """
    Generates image sequences autoregressively using a neural network.

    Args:
        network: The neural network model for generation.
        sampler: The sampling strategy (e.g., TopKSampler).
        sequence_adapter: An instance of GPTAdapter for token interleaving.
        burnin_visual_tokens: Visual tokens for burnin frames, shape [B, T_burnin, H, W].
        burnin_action_tokens: Action tokens for burnin frames, shape [B, T_burnin, K].
        future_action_tokens: Action tokens for future frames, shape [B, T_future, K].
        max_rolling_context_frames: Maximum number of frames to keep in the rolling context.
        temperature: Controls randomness in sampling (default: 1.0).
        return_logits: Whether to return logits along with generated images (default: False).
        verbose: Whether to show progress bars during generation (default: False).

    Returns:
        A dictionary containing generated visual tokens and optionally logits.
    """
    from tqdm.auto import tqdm

    B, nb_burnin_frames, H, W = burnin_visual_tokens.shape
    _, nb_future_actions, nb_action_tokens = future_action_tokens.shape

    # first action conditioning future frame generation is actually already in the context
    nb_future_frames = nb_future_actions + 1

    nb_visual_tokens = H * W

    # Calculate the maximum number of tokens in the rolling context
    max_rolling_context_size = max_rolling_context_frames * (nb_visual_tokens + nb_action_tokens)

    # Interleave burnin visual and action tokens
    sequence_data = sequence_adapter(burnin_visual_tokens, burnin_action_tokens)
    rolling_context = sequence_data['token_sequence']

    # Compute initial positions for the full context size
    position_data = sequence_adapter.compute_position_indices(
        B, H, W, nb_action_tokens, max_rolling_context_frames
    )
    spatial_positions = position_data['spatial_positions']
    temporal_positions = position_data['temporal_positions']

    # Ensure initial context doesn't exceed max_rolling_context_size
    if rolling_context.shape[1] > max_rolling_context_size:
        rolling_context = rolling_context[:, -max_rolling_context_size:]
        spatial_positions = spatial_positions[:, :max_rolling_context_size]
        temporal_positions = temporal_positions[:, :max_rolling_context_size]

    generated_images = []
    generated_logits = []

    # install KV cache in all the Attention layers
    for block in network.transformer.h:
        if not use_kv_cache:
            continue
        layer_dtype = block.attn.c_attn.weight.dtype
        layer_device = block.attn.c_attn.weight.device
        # total_len = rolling_context.shape[1] + nb_future_frames * nb_visual_tokens
        block.attention.cache = KVCache(
            batch_size=B,
            seq_length=max_rolling_context_size,  # this may be larger than the actual context
            n_kv_heads=block.attn.nb_heads,
            head_dim=block.attn.dim_model // block.attn.nb_heads,
            dtype=layer_dtype,
            device=layer_device,
        )

    # Create frame-level progress bar if verbose
    frame_iterator = range(nb_future_frames)
    if verbose:
        frame_iterator = tqdm(frame_iterator, desc='Generating frames')

    prev_pos = 0
    cur_pos = rolling_context.shape[1]  # WARNING: this assumes that all context are the same size for now
    for i in frame_iterator:
        frame_logits = []

        # change of frame, resetting position indices
        rolling_spatial_positions = spatial_positions
        rolling_temporal_positions = temporal_positions

        # Create token-level progress bar if verbose
        token_iterator = range(nb_visual_tokens)
        if verbose:
            token_iterator = tqdm(
                token_iterator,
                desc=f'Frame {i+1}/{nb_future_frames} tokens',
                leave=False)  # Don't leave inner progress bar

        for j in token_iterator:
            # Spatial and temporal pos are pre-computed at len of max_rolling_context_size
            # We need to crop by current_context_len the spatial and temporal pos indices
            # as we are filling the context up to max_rolling_context_size.
            # Once max_rolling_context_size, it will have no effect as rolling logic will take over
            current_context_len = rolling_context.shape[1]

            # Get logits for the next token
            if use_kv_cache:
                sequence_logits = network(
                    rolling_context[:, prev_pos: cur_pos],  # only the last token as the other tokens are cached
                    rolling_spatial_positions[:, current_context_len - 1:current_context_len],
                    rolling_temporal_positions[:, current_context_len - 1:current_context_len],
                    inference=True,
                    start_pos=prev_pos,
                )
                prev_pos = cur_pos
                cur_pos += 1
            else:
                sequence_logits = network(
                    rolling_context,
                    rolling_spatial_positions[:, :current_context_len],
                    rolling_temporal_positions[:, :current_context_len],
                    inference=True,
                )

            next_token_logits = sequence_logits[:, -1, :]  # shape [B, vocab_size]

            if return_logits:
                frame_logits.append(next_token_logits)

            # Apply temperature and sample
            next_token_logits = next_token_logits / temperature
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            next_token = sampler.sample(next_token_probs)  # shape [B, 1]

            # Update rolling context
            rolling_context = torch.cat([rolling_context, next_token], dim=1)
            if rolling_context.shape[1] > max_rolling_context_size:
                rolling_context = rolling_context[:, 1:]
                rolling_spatial_positions = rolling_spatial_positions.roll(-1, dims=1)
                rolling_temporal_positions = rolling_temporal_positions.roll(-1, dims=1)
                rolling_temporal_positions[:, -1] += max_rolling_context_frames

        # Extract the last generated frame from the rolling context and save it
        last_generated_image = rolling_context[:, -nb_visual_tokens:]
        last_generated_image = rearrange(last_generated_image, 'b (h w) -> b h w', h=H)
        generated_images.append(last_generated_image)

        if return_logits:
            frame_logits = torch.stack(frame_logits, dim=1)  # shape [B, H*W, vocab_size]
            frame_logits = rearrange(frame_logits, 'b (h w) v -> b h w v', h=H)
            generated_logits.append(frame_logits)

        if i + 1 == nb_future_frames:
            # If last frame, exit generation loop
            break

        # Add next action tokens to rolling context
        next_action_tokens = future_action_tokens[:, i]  # shape [B, nb_action_tokens]
        # Apply shifts to action token indices based on visual and action vocabularies' sizes.
        # Otherwise action and visual tokens may have the same index
        next_action_tokens = sequence_adapter.action_shifts + next_action_tokens
        rolling_context = torch.cat([rolling_context, next_action_tokens], dim=1)

        if rolling_context.shape[1] > max_rolling_context_size:
            rolling_context = rolling_context[:, nb_action_tokens:]

    # clean up the KV cache in all the layers
    for block in network.transformer.h:
        block.attention.cache = None

    # Stack generated images
    generated_images = torch.stack(generated_images, dim=1)
    out_dict = {
        'visual_tokens': generated_images,
        'sequence_data': sequence_data,
        'position_data': position_data
    }

    if return_logits:
        generated_logits = torch.stack(generated_logits, dim=1)
        out_dict['visual_logits'] = generated_logits

    return out_dict
