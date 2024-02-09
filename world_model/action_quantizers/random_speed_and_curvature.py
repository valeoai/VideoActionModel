import torch
import torch.nn as nn

class RandSpeedAndCurvatureTokens(nn.Module):
    
    def __init__(self, speed_vocab_size, curvature_vocab_size):
        super().__init__()
        
        self.speed_vocab_size = speed_vocab_size
        self.curvature_vocab_size = curvature_vocab_size
    
    def forward(self, ego_to_world_rot, ego_to_world_tran, timestamps, **kwargs):
        b, t, *_ =  ego_to_world_rot.shape
        
        speed_tokens = torch.randint(
            low=0, 
            high=self.speed_vocab_size, 
            size=(b,t), 
            device=timestamps.device
        )
        curvature_tokens = torch.randint(
            low=0, 
            high=self.curvature_vocab_size,
            size=(b,t), 
            device=timestamps.device
        )
        
        action_tokens = torch.stack([speed_tokens, curvature_tokens], dim=-1)
        
        return action_tokens