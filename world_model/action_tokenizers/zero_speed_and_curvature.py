import torch
import torch.nn as nn

class ZeroSpeedAndCurvatureTokens(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, ego_to_world_rot, ego_to_world_tran, timestamps, **kwargs):
        b, t, *_ =  ego_to_world_rot.shape
        
        speed_tokens = torch.zeros(
            size=(b,t-1), 
            device=timestamps.device,
            dtype=torch.int64
        )
        curvature_tokens = torch.zeros(
            size=(b,t-1), 
            device=timestamps.device,
            dtype=torch.int64
        )
        
        action_tokens = torch.stack([speed_tokens, curvature_tokens], dim=-1)
        
        return action_tokens