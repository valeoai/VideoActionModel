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


class MultiTokenTopKSampler(TopKSampler):

    def sample(self, next_token_probabilities: torch.Tensor) -> torch.Tensor:
        nb_visual_tokens = next_token_probabilities.size(1)
        next_token_probabilities = rearrange(next_token_probabilities, 'b n v -> (b n) v')
        next_token = super().sample(next_token_probabilities)
        next_token = rearrange(next_token, '(b n) 1 -> b n 1', n=nb_visual_tokens)
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
                continue

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


@torch.no_grad()
def autoregressive_image_multitoken_generation(
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
    verbose = int(verbose)

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

    generated_images = torch.empty(B, nb_future_frames, H, W, device=burnin_visual_tokens.device)
    if return_logits:
        generated_logits = torch.empty(B, nb_future_frames, H, W, network.vocab_size, device=burnin_visual_tokens.device)

    # Create frame-level progress bar if verbose > 0, leave=True if verbose > 1
    for i in tqdm(range(nb_future_frames), desc='Generating frames', disable=verbose < 1, leave=verbose > 1):
        # change of frame, resetting position indices
        rolling_spatial_positions = spatial_positions
        rolling_temporal_positions = temporal_positions

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

        # remove action tokens from the logits
        next_frame_logits = sequence_logits[:, :nb_visual_tokens, :]  # shape [B, nb_visual_tokens, vocab_size]

        # Apply temperature and sample
        next_frame_logits = next_frame_logits / temperature
        next_frame_probs = F.softmax(next_frame_logits, dim=-1)
        next_frame_probs = rearrange(next_frame_probs, 'b n v -> (b n) v')
        next_frame = sampler.sample(next_frame_probs)  # shape [B * nb_visual_tokens, 1]
        next_frame = rearrange(next_frame, '(b n) 1 -> b n', n=nb_visual_tokens)  # shape [B, nb_visual_tokens]

        # Save generated frame and logits
        generated_images[:, i] = rearrange(next_frame, 'b (h w) -> b h w', h=H)
        if return_logits:
            generated_logits[:, i] = rearrange(next_frame_logits, 'b (h w) v -> b h w v', h=H)

        if i + 1 == nb_future_frames:
            # If last frame, exit generation loop
            continue

        # Add next action tokens to rolling context
        next_action_tokens = future_action_tokens[:, i]  # shape [B, nb_action_tokens]
        # Apply shifts to action token indices based on visual and action vocabularies' sizes.
        # Otherwise action and visual tokens may have the same index
        next_action_tokens = sequence_adapter.action_shifts + next_action_tokens
        # Update rolling context with generated frame and next action tokens
        rolling_context = torch.cat([rolling_context, next_frame, next_action_tokens], dim=1)

        # Update rolling context
        if rolling_context.shape[1] > max_rolling_context_size:
            rolling_context = rolling_context[:, nb_visual_and_action_tokens:]
            # In this case spatial positions are fixed
            # rolling_spatial_positions = rolling_spatial_positions.roll(-nb_visual_and_action_tokens, dims=1)
            rolling_temporal_positions = rolling_temporal_positions.roll(-nb_visual_and_action_tokens, dims=1)
            rolling_temporal_positions[:, -nb_visual_and_action_tokens:] += max_rolling_context_frames  # TODO: check this

    # Stack generated images
    out_dict = {
        'visual_tokens': generated_images,
        'sequence_data': sequence_data,
        'position_data': position_data
    }

    if return_logits:
        out_dict['visual_logits'] = generated_logits

    return out_dict


if __name__ == '__main__':
    # temporary test code
    import yaml
    from pathlib import Path

    import torch
    import mup
    from omegaconf import OmegaConf
    from hydra.utils import instantiate

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_logs_dir = Path('~/iveco/scratch_iveco/world_model_JZGC4/').expanduser()
    path_to_worldmodel_repo = Path('~/shared/eramzi/NextTokenPredictor').expanduser()
    model_name = 'muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0217_lr0.0047_0903_1345_1725363936'

    config_file_path = base_logs_dir / 'model_logs_and_checkpoints' / model_name / 'tensorboard/version_0/hparams.yaml'
    with open(config_file_path, 'r') as file:
        training_logged_config = yaml.safe_load(file)
    training_logged_config['model']['network']['multiple_tokens_inference'] = True
    training_logged_config = OmegaConf.create(training_logged_config)

    model_config = training_logged_config.model
    model_config.mup_base_shapes = str(path_to_worldmodel_repo / 'mup_shapes/gpt2_24layers_nobias_basewidth128.bsh')

    network = instantiate(model_config.network)
    mup.set_base_shapes(network, base=model_config.mup_base_shapes)
    sequence_adapter = instantiate(model_config.sequence_adapter)
    action_tokenizer = instantiate(model_config.action_tokenizer)

    network = network.to(device)
    sequence_adapter = sequence_adapter.to(device)
    action_tokenizer = action_tokenizer.to(device)

    network.eval()
    sequence_adapter.eval()
    action_tokenizer.eval()

    sampler = TopKSampler(k=5)

    window_size = 23
    burnin_visual_tokens = torch.randint(0, 1000, (3, 18, 16, 32), device=device)

    b, burnin_size, h, w = burnin_visual_tokens.shape
    future_size = window_size - burnin_size

    burnin_action_tokens = action_tokenizer(burnin_visual_tokens)
    future_action_tokens = torch.zeros((b, future_size - 1, 2), device=device, dtype=torch.long)

    generated_data = autoregressive_image_multitoken_generation(
        network,
        sampler,
        sequence_adapter,
        burnin_visual_tokens,
        burnin_action_tokens,
        future_action_tokens,
        max_rolling_context_frames=network.nb_timesteps - 1,
        temperature=1.0,
        return_logits=True,
        verbose=2,
    )

    print(generated_data['visual_tokens'].shape)
    print(generated_data['visual_logits'].shape)
