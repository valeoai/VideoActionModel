from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor

Kwargs = Dict[str, Any]


class ZeroSpeedAndCurvatureTokens(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, visual_tokens: Tensor, **kwargs: Kwargs) -> Tensor:
        b, t, *_ = visual_tokens.shape

        speed_tokens = torch.zeros(size=(b, t), device=visual_tokens.device, dtype=torch.int64)
        curvature_tokens = torch.zeros(size=(b, t), device=visual_tokens.device, dtype=torch.int64)

        action_tokens = torch.stack([speed_tokens, curvature_tokens], dim=-1)

        return action_tokens
