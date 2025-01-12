import gc
import torch
from collections import defaultdict
from lightning.pytorch.utilities import move_data_to_device
from functools import partial
import numpy as np


import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def generate_coord_plot_data(model_init, run_forward, dataloader, widths, num_backprop_iters, device):
    log_dict = {}
    
    for width in widths:
        
        coord_check_dict = defaultdict(list)
        
        model = model_init(width=width)
        
        model.to(device)
        model.train()

        optimizer = model.configure_optimizers(weight_decay=1e-8, lr=0.001)
        
        for batch_idx, batch in enumerate(dataloader):
            
            batch = move_data_to_device(batch, device)
        
            coord_check_dict_iter = defaultdict(list)
            def hook(module, input, output, key):
                with torch.no_grad():
                    coord_check_dict_iter[key].append(output.abs().mean().item())
        
            coord_check_handles = []
            
            for module_name, module in model.named_modules():
                if module_name == 'vocab_embed':
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='token_embedding')))
                elif module_name== 'spatial_pos_embed':
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='spatial_embedding')))
                elif module_name.endswith('.query'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='query')))
                elif module_name.endswith('.key'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='key')))
                elif module_name.endswith('.value'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='value')))
                elif module_name.endswith('.attn_out'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='attn')))
                elif module_name.endswith('.ffn_down'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='ffn')))
                elif module_name.endswith('.ffn_gate'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='ffn_gate')))
                elif module_name.endswith('.ffn_up'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='ffn_up')))
                elif module_name.endswith('.out'):
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='transformer_block')))
                elif module_name == 'out_proj.1':
                    coord_check_handles.append(module.register_forward_hook(partial(hook, key='output_logits')))
        
            loss = run_forward(batch, model)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            for handle in coord_check_handles:
                handle.remove()
            
            for key, value in coord_check_dict_iter.items():
                coord_check_dict[key].append(np.mean(value))
                
            if batch_idx == num_backprop_iters - 1: 
                break
        
        log_dict[width] = coord_check_dict
    
        del model
        torch.cuda.empty_cache()    
        gc.collect()
        torch.cuda.empty_cache()
    
    return log_dict    

def restructure_log_dict(log_dict):
    # Get all layer types from first width (since all widths have same layer types)
    first_width = list(log_dict.keys())[0]
    layer_types = log_dict[first_width].keys()
    
    # Get number of iterations (since all layer types have same number of iterations)
    first_layer = list(layer_types)[0]
    nb_iters = len(log_dict[first_width][first_layer])
    
    result = {}
    
    for layer_type in layer_types:
        result[layer_type] = {}
        for iter_idx in range(nb_iters):
            result[layer_type][iter_idx] = {
                'widths': [],
                'means': []
            }
            for width in log_dict.keys():
                result[layer_type][iter_idx]['widths'].append(width)
                result[layer_type][iter_idx]['means'].append(log_dict[width][layer_type][iter_idx])
    
    return result