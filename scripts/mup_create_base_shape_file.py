from world_model.networks.mu_gpt2 import MuGPT2

from mup import make_base_shapes


if __name__ == "__main__":
    
    nb_layers = 12
    nb_timesteps = 16
    dim_per_head = 64
    
    width = 128
    base_model = MuGPT2(
        embedding_dim=width,
        nb_layers=nb_layers,
        nb_heads=width//dim_per_head, 
        vocabulary_size=1024+50+80, 
        nb_timesteps=nb_timesteps, 
        nb_tokens_per_timestep=352, 
        dropout_rate=0.,
    )

    # The delta model is used to automatically detect "infinite" dimensions.
    # That is to say the weights matrices that changes/scales with hyper-params
    # and requiring to be properly scaled.
    width = 256
    delta_model = MuGPT2(
        embedding_dim=width,
        nb_layers=nb_layers,
        nb_heads=width//dim_per_head, 
        vocabulary_size=1024+50+80, 
        nb_timesteps=nb_timesteps, 
        nb_tokens_per_timestep=352, 
        dropout_rate=0.,
    )
    
    file_path = '../mup_shapes/gpt2_12layers_16timesteps_basewidth128.bsh'
    base_shapes = make_base_shapes(base_model, delta_model, savefile=file_path)
