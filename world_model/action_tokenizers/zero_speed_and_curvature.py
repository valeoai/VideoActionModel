import torch
import torch.nn as nn

class ZeroSpeedAndCurvatureTokens(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, visual_tokens, **kwargs):
        b, t, *_ =  visual_tokens.shape
        
        speed_tokens = torch.zeros(
            size=(b,t-1), 
            device=visual_tokens.device,
            dtype=torch.int64
        )
        curvature_tokens = torch.zeros(
            size=(b,t-1), 
            device=visual_tokens.device,
            dtype=torch.int64
        )
        
        action_tokens = torch.stack([speed_tokens, curvature_tokens], dim=-1)
        
        return action_tokens