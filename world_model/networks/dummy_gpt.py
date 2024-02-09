import torch
import torch.nn as nn
from einops import rearrange

class DummyGPT(nn.Module):
    
    def __init__(
        self, 
        embedding_dim,
        num_heads,
        num_layers, 
        vocabulary_size,    
        nb_timesteps, # not used in DummyGPT
        nb_tokens_per_timestep # not used in DummyGPT
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predictor = nn.Linear(embedding_dim, vocabulary_size)
        
    def forward(self, tokens_sequence):
        embedding_sequence = self.embedding(tokens_sequence)
        
        processed_sequence = self.transformer_encoder(embedding_sequence)
        
        logits_sequence = self.predictor(processed_sequence)
        
        return logits_sequence
    
    def inference_forward(self, tokens_sequence):
        embedding_sequence = self.embedding(tokens_sequence)
        
        processed_sequence = self.transformer_encoder(embedding_sequence)
        
        # Linear layer only need to be applied on last token
        logits_sequence = self.predictor(processed_sequence[:,-1,:])
        
        return logits_sequence