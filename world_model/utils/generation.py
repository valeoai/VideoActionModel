import torch
import torch.nn.functional as F
from typing import Any
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
    context_visual_tokens,
    context_action_tokens,
    future_action_tokens,
    temperature = 1.0,
    return_logits = False
):

    """
    temperature: controls the diversity/predictability of the output:
        - lower temperature sharpens the probability distribution -> more predictable results.
        - higher temperature flattens it -> increases chance of sampling less likely outcomes.
        
    """
    
    _, t, h, w = context_visual_tokens.shape
    _, n, k = future_action_tokens.shape

    nb_context_frames = t
    nb_future_frames = n + 1
    nb_visual_tokens = h * w
    nb_action_tokens = k

    sequence_data = sequence_adapter(context_visual_tokens, context_action_tokens)
    
    generated_images = []
    generated_logits = []
    
    rolling_context = sequence_data['token_sequence']

    with torch.no_grad():
        for i in range(nb_future_frames):
            
            frame_logits = []
            
            rolling_spatial_positions = sequence_data['spatial_positions']
            rolling_temporal_positions = sequence_data['temporal_positions']
            
            for _ in range(nb_visual_tokens):

                # logits shape [B, seq_len, vocab_size]
                sequence_logits = network(
                    rolling_context,
                    rolling_spatial_positions,
                    rolling_temporal_positions
                )
                
                next_token_logits = sequence_logits[:, -1, :] # shape [B, vocab_size]
                
                if return_logits:
                    frame_logits.append(next_token_logits)
                
                next_token_logits = next_token_logits / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                next_token = sampler.sample(next_token_probs) # shape [B, 1]
                
                rolling_context = torch.cat((rolling_context[:, 1:], next_token), dim=-1)

                # Update spatial positions: Just rotate left by 1 for visual tokens
                rolling_spatial_positions = torch.cat([
                    rolling_spatial_positions[:, 1:],
                    rolling_spatial_positions[:, :1]
                ], dim=-1)

                # Update temporal positions: Rotate left by 1 and shift index value by the window size
                rolling_temporal_positions = torch.cat([
                    rolling_temporal_positions[:, 1:],
                    rolling_temporal_positions[:, :1] + nb_context_frames
                ], dim=-1)

            # Extract the last generated frame from the rolling context and save it
            last_generated_image = rolling_context[:,-nb_visual_tokens:]
            last_generated_image = rearrange(last_generated_image, 'b (h w) -> b h w', h=h)
            generated_images.append(last_generated_image)
            
            if return_logits:
                frame_logits = torch.stack(frame_logits, dim=1) # shape [B, H*W, vocab_size]
                frame_logits = rearrange(frame_logits, 'b (h w) v -> b h w v', h=h)
                generated_logits.append(frame_logits)

            if i+1 == nb_future_frames:
                # if last frame we have finished, get out of generation loop
                break
                
            # Add GT action to rolling context to continue with the next frame generation
            next_action_tokens = future_action_tokens[:,i] # shape [B, nb_action_tokens]
            rolling_context = torch.cat([
                rolling_context[..., nb_action_tokens:], # Rotate left by `nb_action_tokens`
                next_action_tokens
            ], dim=-1)
            
    
    # generated_images is a list of t tensors of shape [B, H, W]    
    generated_images = torch.stack(generated_images, dim=1)    
    out_dict = {'visual_tokens': generated_images}
    if return_logits:
        # generated_logits is a list of t tensors of shape [B, H, W, vocab_size]
        generated_logits = torch.stack(generated_logits, dim=1)
        out_dict['visual_logits'] = generated_logits

    return out_dict