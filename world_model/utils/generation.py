import torch
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

    with torch.no_grad():
        # Create frame-level progress bar if verbose
        frame_iterator = range(nb_future_frames)
        if verbose:
            frame_iterator = tqdm(frame_iterator, desc='Generating frames')

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


def autoregressive_image_frame_generation(
    network,
    sampler,
    sequence_adapter,
    burnin_visual_tokens,
    burnin_action_tokens,
    future_action_tokens,
    max_rolling_context_frames,
    temperature=1.0,
    return_logits=False,
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
    nb_visual_and_action_tokens = nb_visual_tokens + nb_action_tokens

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

    with torch.no_grad():
        # Create frame-level progress bar if verbose
        frame_iterator = range(nb_future_frames)
        if verbose:
            frame_iterator = tqdm(frame_iterator, desc='Generating frames')

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

            # Spatial and temporal pos are pre-computed at len of max_rolling_context_size
            # We need to crop by current_context_len the spatial and temporal pos indices
            # as we are filling the context up to max_rolling_context_size.
            # Once max_rolling_context_size, it will have no effect as rolling logic will take over
            current_context_len = rolling_context.shape[1]

            # Get logits for the next token
            sequence_logits = network(
                rolling_context,
                rolling_spatial_positions[:, :current_context_len],
                rolling_temporal_positions[:, :current_context_len],
                inference=True,
            )

            next_frame_logits = sequence_logits[:, -nb_visual_tokens, :]  # shape [B, nb_visual_tokens, vocab_size]

            if return_logits:
                frame_logits.append(next_frame_logits)

            # Apply temperature and sample
            next_frame_logits = next_frame_logits / temperature
            next_frame_probs = F.softmax(next_frame_logits, dim=-1)
            next_frame = sampler.sample(next_frame_probs)  # shape [B, 1]

            # Update rolling context
            rolling_context = torch.cat([rolling_context, next_frame], dim=1)
            if rolling_context.shape[1] > max_rolling_context_size:
                rolling_context = rolling_context[:, nb_visual_and_action_tokens:]
                rolling_spatial_positions = rolling_spatial_positions.roll(-nb_visual_and_action_tokens, dims=1)
                rolling_temporal_positions = rolling_temporal_positions.roll(-nb_visual_and_action_tokens, dims=1)
                rolling_temporal_positions[:, -1] += max_rolling_context_frames  # TODO: check this

            # Extract the last generated frame from the rolling context and save it
            generated_images.append(rearrange(next_frame, 'b (h w) -> b h w', h=H))

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
