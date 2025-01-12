import torch
import torch.nn as nn
from torch import Tensor
from layers import TransformerLayer
from pos_embedding import precompute_freqs_cis
from collections import defaultdict

class ActionExpert(nn.Module):

    base_width = 256
    
    def __init__(
        self, 
        attn_dim: int,
        ffn_dim: int,
        depth:int,
        head_dim: int, 
        horizon_lenght: int,
        ffn_expansion: int,
        nb_timesteps: int,
        rope_theta: float,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()

        self.attn_dim = attn_dim
        self.ffn_dim = ffn_dim
        self.depth = depth
        self.ffn_expansion = ffn_expansion
        self.nb_timesteps = nb_timesteps
        self.init_std = init_std
        
        self.spatial_pos_embed = nn.Embedding(horizon_lenght, ffn_dim)
        self.norm = nn.RMSNorm(ffn_dim, elementwise_affine=False),

        self.layers = nn.Sequential(*(
            TransformerLayer(
                attn_dim, 
                head_dim, 
                ffn_expansion,
                layer_idx,
                ffn_dim
            ) for layer_idx in range(depth)
        ))

        self.freqs_cis = precompute_freqs_cis(
            head_dim,
            nb_timesteps,
            rope_theta,
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        '''
        In addition to muP we apply special scaled init to the residual projections (attn & mlp), per GPT-2 paper
        
            > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            
        This is combined with the muP initilization that is scaling init_std by sqrt(base_fan_in/fan_in).
        Note: there are 2 * self.depth # of residual layers because each transformer block has res attn and res mlp
        '''
        
        # muP init
        # Note that if attention uses nn.Linear fanout is first dim
        # because nn.Linear applies y=xA.T+b
        # if using Conv1D as in hugginfaces transformers, fanout is last dim
        
        if isinstance(module, nn.Embedding):
            # Embeddings uses init_std directly
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.init_std)
        
        if isinstance(module, TransformerLayer):

            # init all modules where base_fan_in = base_width
            base_fan_in = self.base_width
            fan_in = module.attn_out.weight.shape[1]
            std = self.init_std * (base_fan_in / fan_in) ** 0.5
            torch.nn.init.normal_(module.attn_qkv.weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.ffn_up.weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.ffn_gate.weight, mean=0.0, std=std) 

            ### muP zero-initialized query projections
            # https://arxiv.org/abs/2404.05728
            # yield equal attention weights over all past timesteps at initialization
            # if attention uses nn.Linear change init in first dim
            # if using Conv1D, change init in last dim
            fanout, _ = module.attn_qkv.weight.shape
            assert fanout % 3 == 0  # assert matrix is used for query, key and value
            module.attn_qkv.weight.data[:fanout // 3, :] = 0

            # init all modules where base_fan_in != base_width
            base_fan_in = self.base_width * self.ffn_expansion
            fan_in = module.ffn_down.weight.shape[1] 
            std = self.init_std * (2 * self.depth * base_fan_in / fan_in) ** 0.5
            torch.nn.init.normal_(module.ffn_down.weight, mean=0.0, std=std)    
                     

    def configure_optimizers(self, weight_decay, lr, *args, **kwargs):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        groups = defaultdict(list)

        ### Begin muP code ###
        for n, p in param_dict.items():
            if p.dim() < 2:
                # no weight decay, no lr scaling
                groups[(0.0, lr)].append(p)
                continue

            # for weights with weight decay we use decoupling
            # weight_decay/learning_rate
            # Note: only independent of peak LR, not of schedule

            hidden_list = ['attn_qkv', 'attn_out', 'ffn_up', 'ffn_gate', 'ffn_down']

            is_mup = False
            for mup_weight in hidden_list:
                if n.endswith(mup_weight+'.weight'):
                    is_mup = True

            if is_mup:
                fan_in = p.shape[1]
                base_factor = fan_in / self.width # handle layers with width expansion
                scaled_lr = lr * self.base_width * base_factor / fan_in
                groups[(weight_decay/scaled_lr, scaled_lr)].append(p)
            else:
                groups[(weight_decay/lr, lr)].append(p)
            
        ### End muP code ###

        i = 0
        for (wd, lr), params in groups.items():
            num_params = sum(p.numel() for p in params)
            print(f"OPTIM GROUP {i} | wd: {wd:.4f} | lr {lr:.4f} | num parameter tensors: {len(params)}, with {num_params:,} parameters")
            i += 1

        optim_groups = [{'params': params, 'weight_decay': wd, 'lr': lr} for (wd, lr), params in groups.items()]
        
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, weight_decay=weight_decay,  *args, **kwargs)

        return optimizer

    def forward(
        self,
        hidden_states,
        spatial_positions,
        temporal_positions,
        inference=False
    ) -> Tensor:
        
        """
        Args:
            spatial_positions: A tensor indicating the spatial position of each token in the sequence.
                example: [0,1,2,3,0,1,2,3]
            temporal_positions: A tensor indicating the temporal position of each token in the sequence.
                example: [0,0,0,0,1,1,1,1]
        """
        assert spatial_positions.max() < self.nb_tokens_per_timestep, f"spatial_positions.max()={spatial_positions.max()} >= self.nb_tokens_per_timestep={self.nb_tokens_per_timestep}"
        assert temporal_positions.max() < self.nb_timesteps, f"temporal_positions.max()={temporal_positions.max()} >= self.nb_timesteps={self.nb_timesteps}"

        # compute spatial position embeddings
        spatial_pos_emb = self.spatial_pos_embed(spatial_positions)
        
        hidden_states = hidden_states + spatial_pos_emb
        hidden_states = self.norm(hidden_states)
        
        seqlen = hidden_states.size(1)
        attn_mask = torch.tril(torch.ones(seqlen, seqlen, device=hidden_states.device, dtype=torch.bool))
        attn_mask = attn_mask.type_as(hidden_states)

        self.freqs_cis = self.freqs_cis.to(spatial_pos_emb.device)
        freqs_cis = self.freqs_cis[temporal_positions]
        
        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attn_mask=attn_mask,
                freqs_cis=freqs_cis,
            )
        
        return hidden_states