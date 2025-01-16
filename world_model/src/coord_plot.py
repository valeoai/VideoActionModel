import gc
from collections import defaultdict
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.utilities import move_data_to_device
from matplotlib import cm


class MplColorHelper:

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def generate_coord_plot_data(
    track_list, model_init, run_forward, dataloader, widths, num_backprop_iters, device, optimizer=None
):
    log_dict = {}

    for width in widths:

        coord_check_dict = defaultdict(list)

        model, optimizer = model_init(width=width)

        model.to(device)
        model.train()

        for batch_idx, batch in enumerate(dataloader):

            batch = move_data_to_device(batch, device)

            coord_check_dict_iter = defaultdict(list)

            def hook(module, input, output, key):
                with torch.no_grad():
                    coord_check_dict_iter[key].append(output.abs().mean().item())

            coord_check_handles = []

            for module_name, module in model.named_modules():
                for track_type, track_pattern, key_name in track_list:
                    if track_type == "equals":
                        if module_name == track_pattern:
                            coord_check_handles.append(module.register_forward_hook(partial(hook, key=key_name)))
                            break
                    elif track_type == "endswith":
                        if module_name.endswith(track_pattern):
                            coord_check_handles.append(module.register_forward_hook(partial(hook, key=key_name)))
                            break
                    else:
                        raise ValueError("track_type '{track_type}' not in ['equals', 'endswith']")

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


track_list_llama = [  # TODO: messed things up need to redo it
    ("equals", "vocab_embed", "token_embedding"),
    ("equals", "spatial_pos_embed", "spatial_embedding"),
    ("equals", "out_proj.1", "output_logits"),
    ("endswith", ".attn_out", "attn"),
    ("endswith", ".query", "query"),
    ("endswith", ".key", "key"),
    ("endswith", ".value", "value"),
    ("endswith", ".attn.c_proj", "attn_out"),
    ("endswith", ".ffn_down", "ffn"),
    ("endswith", ".ffn_gate", "ffn_gate"),
    ("endswith", ".ffn_up", "ffn_up"),
    ("endswith", ".out", "transformer_block"),
]

track_list_action_expert = [
    ("equals", "action_encoder.command_embedding", "command_embedding"),
    ("equals", "action_encoder.action_positional_embedding", "spatial_embedding"),
    ("equals", "action_encoder.linear_1", "action_embed"),
    ("equals", "action_encoder.linear_2", "action+diffusion_step"),
    ("equals", "action_encoder.linear_3", "action_encoder_out"),
    ("endswith", ".c_attn", "attn_in"),
    ("endswith", ".attn.c_proj", "attn_out"),
    ("endswith", ".query", "query"),
    ("endswith", ".key", "key"),
    ("endswith", ".value", "value"),
    ("endswith", ".mlp.c_fc", "ffn_in_proj"),
    ("endswith", ".mlp.c_proj", "ffn_out_proj"),
    ("equals", "action_decoder", "action_decoder_out"),
]


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
            result[layer_type][iter_idx] = {"widths": [], "means": []}
            for width in log_dict.keys():
                result[layer_type][iter_idx]["widths"].append(width)
                result[layer_type][iter_idx]["means"].append(log_dict[width][layer_type][iter_idx])

    return result
