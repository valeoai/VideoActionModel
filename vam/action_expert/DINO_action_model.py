import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import mup
import torch
import torch.nn as nn
from einops import rearrange
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import Tensor, LongTensor
from tqdm import tqdm

# ------------------------------------------------------------------------------------
# Vision backbone : DINOv2
# ------------------------------------------------------------------------------------

try:
    import timm  # type: ignore
except ImportError as err:  # pragma: no cover
    raise ImportError(
        "timm is required for the DINOv2 baseline – install with `pip install timm`"
    ) from err


class DINOBackbone(nn.Module):
    """Frozen DINO‑v2 ViT that returns *patch* embeddings for every video frame.

    Returned tensor shape: **[B, T * N, D]**, where *N* is the number of spatial
    patches per frame and T the number of frames.  The model stays in `eval()` mode and `requires_grad_(False)`
    by default
    """

    def __init__(self, model_name: str = "vit_large_patch14_dinov2.lvd142m") -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=True)
        self.patch_size = self.vit.patch_embed.patch_size  # (14, 14) for L/14
        self.embedding_dim = self.vit.embed_dim  # 1024 for L/14
        self.vit.eval()
        self.vit.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:  # x : [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = rearrange(x, "b t c h w -> (b t) c h w")
        # timm's forward_features:   cls | patch_0 … patch_N
        feats = self.vit.forward_features(x)  # [B*T, 1+N, D]
        patch_tokens = feats[:, 1:]  # drop CLS
        patch_tokens = rearrange(patch_tokens, "(b t) n d -> b (t n) d", b=b, t=t)
        return patch_tokens


# ------------------------------------------------------------------------------------
# Joint model: static DINO tokens  ↔  Action‑Expert transformer
# ------------------------------------------------------------------------------------

from vam.action_expert.mup_action_expert import (
    Block as ActionBlock,
    MupActionExpert,
    SelfAttention as ActionAttention,
)

InputsDict = Dict[str, Tensor]
OutputDict = Dict[str, Tensor]


class JointModelDINO(nn.Module):
    """Action‑expert cross‑attending to *frozen* DINO tokens.

    Added: **frame‑index positional embedding** so the action expert can
    tell early vs. late frames.
    """

    def __init__(
        self,
        dino: DINOBackbone,
        action_expert: MupActionExpert,
        context_length: int,
    ) -> None:
        super().__init__()
        self.dino = dino
        self.action_expert = action_expert
        self.context_length = context_length  # maximum #video frames seen during training

        # dimensions
        self.num_hidden_layers = len(self.action_expert.transformer.h)
        self.visual_dim = self.dino.embedding_dim
        self.action_dim = self.action_expert.embedding_dim  # (=attention dim)

        # Dino Feature Projection
        self.visual_to_action = nn.Sequential(
            nn.Linear(self.visual_dim, self.action_dim, bias=False),
            nn.RMSNorm(self.action_dim, elementwise_affine=False)
        )

        # Learnable frame‑index embedding (D_a sized)
        self.frame_index_embed = nn.Embedding(context_length, self.action_dim)
        
        self._build_block_causal_mask()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _visual_tokens_to_kv(self, visual_tokens: Tensor, layer_idx: int) -> Tuple[Tensor, Tensor]:
        """Project frozen visual tokens to K and V for *one* layer."""
        action_block: ActionBlock = self.action_expert.transformer.h[layer_idx]
        action_attention: ActionAttention = action_block.attn
        
        _, action_k, action_v = rearrange(
            action_attention.c_attn(visual_tokens),
            "b seq (n nb_heads dim_heads) -> n b nb_heads seq dim_heads",
            n=3,
            dim_heads=action_attention.dim_heads,
        )
        
        return action_k, action_v

    def _noisy_action_to_embeds(
        self, noisy_action: Tensor, high_level_command: LongTensor, t: Tensor
    ) -> Tensor:
        """Action embedding of the action expert model."""
        # noisy_action: [Batch_Size, timesteps, Horizon_Steps, Action_Dim]
        action_embeds = self.action_expert.action_encoder(
            actions=noisy_action, high_level_command=high_level_command, diffusion_step=t
        )
        action_embeds = rearrange(action_embeds, "b t h d -> b (t h) d")
        return action_embeds
    
    # ------------------------------------------------------------------
    # block‑causal mask builder
    # ------------------------------------------------------------------

    def _build_block_causal_mask(self, T: int, N: int, H: int, device: torch.device) -> Tensor:
        """
        Create a block‑causal boolean mask for cross‑attention.

        Shape: ``[S_act, S_vis + S_act]`` where
            S_vis = T * N  (visual tokens)
            S_act = T * H  (action tokens)

        * ``True``  – key token is visible to the query
        * ``False`` – key token is hidden

        Order of tokens in the key sequence:
            V0,0 … V0,N‑1, V1,0 … V(T‑1),N‑1, a0,0 … a0,H‑1, a1,0 … a(T‑1),H‑1

        Example with T=2, N=2, H=3 (rows = action queries):

               V0 V0 V1 V1 | a0 a0 a0 a1 a1 a1
            a0  1  1  0  0 | 1  1  1  0  0  0
            a0  1  1  0  0 | 1  1  1  0  0  0
            a0  1  1  0  0 | 1  1  1  0  0  0
            a1  1  1  1  1 | 0  0  0  1  1  1
            a1  1  1  1  1 | 0  0  0  1  1  1
            a1  1  1  1  1 | 0  0  0  1  1  1

        Rule: action tokens at time t see all visual tokens from frames ≤ t and
        their own H action tokens.
        """
        S_vis = T * N

        # timestep index for each action query row
        row_t = torch.repeat_interleave(torch.arange(T, device=device), H)  # [S_act]

        # ----- visual part --------------------------------------------------
        vis_frame_idx = torch.arange(S_vis, device=device) // N  # [S_vis]
        allowed_vis = vis_frame_idx.unsqueeze(0) <= row_t.unsqueeze(1)  # broadcast

        # ----- action part --------------------------------------------------
        act_time_idx = torch.repeat_interleave(torch.arange(T, device=device), H)  # [S_act]
        allowed_act = act_time_idx.unsqueeze(0) == row_t.unsqueeze(1)

        attn_mask = torch.cat([allowed_vis, allowed_act], dim=-1)  # [S_act, S_vis+S_act]
        
        self.register_buffer("attn_mask", attn_mask)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(self, inputs_all: InputsDict, return_visual_embeds: bool = False) -> OutputDict:
        # ---------------- 1) Visual tokens (+ time PE) ---------------
        B, T = inputs_all["video_frames"].shape[0:2]
        visual_embeds = self.dino(inputs_all["video_frames"])  # [B, T, N, D_v]
        visual_embeds = self.visual_to_action(visual_embeds)  # [B, T, N, D_a]

        N_patch = visual_embeds.size(2) // T
        device = visual_embeds.device
        frame_ids = torch.arange(T - 1, device=device).repeat_interleave(N_patch)  # [T·N]
        pe = self.frame_index_embed(frame_ids)  # [T·N, D_a]
        visual_embeds = visual_embeds + pe.unsqueeze(0)  # broadcast over batch

        # ---------------- 2) Action tokens -----------------------------------
        action_embeds = self._noisy_action_to_embeds(
            inputs_all["noisy_actions"], inputs_all["high_level_command"], inputs_all["diffusion_step"]
        )  # [B, S_act, D_a]

        # ---------------- 3) Transformer cross‑attn layers -------------------
        for idx in range(self.num_hidden_layers):
            action_embeds = self._forward_single_layer(visual_embeds, action_embeds, idx)

        # ---------------- 4) Decode back to actions --------------------------
        action_embeds = self.action_expert.transformer.ln_f(action_embeds)
        denoised_actions = self.action_expert.action_decoder(action_embeds)
        denoised_actions = rearrange(
            denoised_actions,
            "b (t h) d -> b t h d",
            t=inputs_all["noisy_actions"].size(1),
            h=self.action_expert.action_horizon,
        )

        out: OutputDict = {
            "actions": denoised_actions,
            "actions_embeds": rearrange(
                action_embeds,
                "b (t h) d -> b t h d",
                t=inputs_all["noisy_actions"].size(1),
                h=self.action_expert.action_horizon,
            ),
        }
        if return_visual_embeds:
            out["visual_embeds"] = visual_embeds
        return out

    # ------------------------------------------------------------------
    # One transformer layer of cross‑attention + FFN
    # ------------------------------------------------------------------

    def _forward_single_layer(self, attn_mask: Tensor, visual: Tensor, action: Tensor, layer_idx: int) -> Tensor:
        blk: ActionBlock = self.action_expert.transformer.h[layer_idx]
        attn: ActionAttention = blk.attn

        # LayerNorm on action branch
        act_in = blk.ln_1(action)
        act_qkv = attn.c_attn(act_in)
        q, k_act, v_act = rearrange(
            act_qkv, "b s (n h d) -> n b h s d", n=3, h=attn.num_heads, d=attn.dim_heads
        )

        # Visual K,V (static) – reshape to match heads dimension
        k_vis, v_vis = self._visual_tokens_to_kv(visual, layer_idx)
        k_vis = rearrange(k_vis, "b s (h d) -> b h s d", h=attn.num_heads, d=attn.dim_heads)
        v_vis = rearrange(v_vis, "b s (h d) -> b h s d", h=attn.num_heads, d=attn.dim_heads)

        k = torch.cat([k_vis, k_act], dim=-2)
        v = torch.cat([v_vis, v_act], dim=-2)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=self.attn_mask, is_causal=False, scale=attn.attn_scale / attn.dim_heads
        )
        y = rearrange(y, "b h s d -> b s (h d)")
        y = attn.c_proj(y)

        # Residual + FFN
        action = action + y
        action = action + blk.mlp(blk.ln_2(action))
        return action


# ------------------------------------------------------------------------------------
# Wrapper: training + inference API (flow‑matching)
# ------------------------------------------------------------------------------------

class DINOActionModel(nn.Module):
    """Flow‑matching diffusion model that uses DINOv2 visual features."""

    def __init__(
        self,
        dino_config: OmegaConf,
        action_config: OmegaConf,
        action_mup_base_shapes: mup.MuReadOnly | None,
        action_checkpoint_path: Optional[str],
        context_length: int = 8,
        num_inference_steps: int = 10,
        flow_sig_min: float = 0.001,
        final_action_clip_value: Optional[float] = None,
        action_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        # params ----------------------------------------------------------------
        self.num_inference_steps = num_inference_steps
        self.flow_sig_min = flow_sig_min
        self.final_action_clip_value = final_action_clip_value
        self.context_length = context_length
        self.action_scaling = action_scaling

        # models ----------------------------------------------------------------
        self.dino: DINOBackbone = instantiate(dino_config)
        self.action_expert: MupActionExpert = instantiate(action_config)

        if action_checkpoint_path is not None:
            sd = torch.load(action_checkpoint_path, map_location="cpu")["state_dict"]
            self.action_expert.load_state_dict(sd, strict=False)
            mup.set_base_shapes(self.action_expert, action_mup_base_shapes, rescale_params=False)
            self.action_expert.requires_grad_(False)
        else:
            mup.set_base_shapes(self.action_expert, action_mup_base_shapes)
            self.action_expert.apply(self.action_expert._init_weights)

        self.joint_model = JointModelDINO(
            self.dino, self.action_expert, context_length=context_length
        )

        # convenience
        self.action_dim = self.action_expert.action_dim
        self.action_horizon = self.action_expert.action_horizon

    # ------------------------------------------------------------------
    # Training : flow‑matching loss
    # ------------------------------------------------------------------

    def psi_t(self, x: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        t = t[:, :, None, None]
        return (1 - (1 - self.flow_sig_min) * t) * x + t * x1

    def forward(
        self,
        video_frames: Tensor,  # [B, T, C, H, W]
        high_level_command: LongTensor,  # [B, T]
        actions: Tensor,  # [B, T, H, A]
        t: Tensor,  # [B, T]
    ) -> Tensor:  # loss
        x0 = torch.randn_like(actions)
        x1 = actions / self.action_scaling
        psi_t = self.psi_t(x0, x1, t.type_as(x1))

        out = self.joint_model(
            {
                "video_frames": video_frames,
                "noisy_actions": psi_t,
                "high_level_command": high_level_command,
                "diffusion_step": t,
            }
        )
        v_psi = out["actions"]
        d_psi = x1 - (1 - self.flow_sig_min) * x0
        return torch.mean((v_psi - d_psi) ** 2)

    # ------------------------------------------------------------------
    # Inference : Euler solver
    # ------------------------------------------------------------------

    def forward_inference(
        self,
        video_frames: Tensor,  # [B, T, C, H, W]
        high_level_command: LongTensor,  # [B, 1]
        dtype: torch.dtype = torch.float32,
        num_inference_steps: Optional[int] = None,
        final_action_clip_value: Optional[float] = None,
        verbose: bool = False,
    ) -> Tensor:  # [B, 1, H, A]
        device = video_frames.device
        B = video_frames.shape[0]
        num_inference_steps = num_inference_steps or self.num_inference_steps
        final_action_clip_value = final_action_clip_value or self.final_action_clip_value

        def _post(x: Tensor) -> Tensor:
            if final_action_clip_value is not None:
                x = torch.clamp(x, -final_action_clip_value, final_action_clip_value)
            return x * self.action_scaling

        action = torch.randn(
            (B, 1, self.action_horizon, self.action_dim), device=device, dtype=dtype
        )
        dt = 1.0 / num_inference_steps
        t = torch.zeros((B, 1), device=device, dtype=dtype)
        for _ in tqdm(range(num_inference_steps), disable=not verbose, desc="Euler int"):
            vel = self.joint_model(
                {
                    "video_frames": video_frames,
                    "noisy_actions": action,
                    "high_level_command": high_level_command,
                    "diffusion_step": t,
                }
            )["actions"]
            action = action + dt * vel
            t = t + dt
        return _post(action)
