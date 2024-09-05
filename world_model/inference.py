"""
This script provides code to generate future frames tokens from a pre-trained world-model. 

Outputs follow this structure: 

/<output_dir>/
    <model_log_dir name>/
        inference/
            <checkpoint_name>/
                <top_k,top_p,argmax>_sampling/
                    generated_tokens_metadata.yaml
                    generation_<index number of generation [0,N-1]>/
                        tokens/
                            samples/
                                CAM_FRONT/
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708429012404.pkl
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708430512404.pkl
                                    ...
                        
                        logits/ 
                            samples/
                                CAM_FRONT/
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708429012404.pkl
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708430512404.pkl
                                    ...
            
With every pickle file under the `tokens` folder containing:
- visual_tokens of shape [H, W]

With every pickle file under the `logits` (only for generation_0) folder containing:
- visual_logits of shape [vocab_size, H, W]

generated_tokens_metadata.yaml contains:
    - inference_config: the configuration and metadata used to run the inference (e.g., starting index
    of the generation in the sequence, number of context and prediction frames, the 
    original path to the model's checkpoint, the inference code git hash ...)
    - training_config: the initial configuration (e.g., hyperparams) and metadata (training code git hash)
    used to train the model.
"""

from typing import Any, Dict, List, Tuple
import pickle
from tqdm import tqdm
import numpy as np
from pathlib import Path
import git
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from einops import rearrange
import sys
import mup

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lightning.pytorch.utilities import move_data_to_device

from world_model.dataloader.components.scene_tokenized_sequence_nuplan import SceneBasedNuplanDataset
from world_model.dataloader.tokenized_sequence_nuplan import custom_collate
from world_model.utils.generation import TopKSampler, autoregressive_image_sequence_generation
from world_model.utils import  RankedLogger, extras, instantiate_samplers

log = RankedLogger(__name__, rank_zero_only=True)

def worker_rnd_init(x):
    np.random.seed(42 + x)
    
def print_log_and_current_config(inference_config, training_logged_config):
    print('ORIGINAL train dataset configuration')
    print('\t pickle_path: \t\t\t', training_logged_config.data.pickle_path)
    print('\t train_pickle_name: \t\t', training_logged_config.data.train_pickle_name)
    print('\t val_pickle_name: \t\t', training_logged_config.data.val_pickle_name)
    print('\t data_root_dir: \t\t', training_logged_config.data.train_dataset_params.data_root_dir)
    print('\t quantized_data_root_dir: \t', training_logged_config.data.train_dataset_params.quantized_data_root_dir)
    print('\t sequence_length: \t\t', training_logged_config.data.train_dataset_params.sequence_length)
    print('\t subsampling_factor: \t\t', training_logged_config.data.train_dataset_params.subsampling_factor)
    print()
    print('CURRENT inference configuration')
    print('\t pickle_path: \t\t\t', inference_config.paths.pickle_path)
    print('\t pickle_name: \t\t\t', inference_config.pickle_name)
    print('\t quantized_data_root_dir: \t', inference_config.paths.quantized_data_root_dir)
    print('\t nb_context_frames: \t\t', inference_config.dataset_config.nb_context_frames)
    print('\t nb_prediction_frames: \t\t', inference_config.dataset_config.nb_prediction_frames)
    print('\t subsampling_factor: \t\t', inference_config.dataset_config.subsampling_factor)

    
def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    """
    Used to remove the "network" prefixes from the state-dict
    We are only loading the network and not the full model (scheduler, optimizer, etc..)
    However, the weights of the network in the checkpoint all have a "network" prefix because network is an attribute of the model.
    We are removing these prefixes from the weights as we are loading the network alone and not the full model.
    """
    
    result = dict()

    for k, v in state_dict.items():
        tokens = k.split('.')

        if tokens[0] == prefix:
            tokens = tokens[1:]

            key = '.'.join(tokens)
            result[key] = v

    return result

def load_model(checkpoint_path, model_config, device, inference_sha, set_net_to_eval=True):
    
    checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
    
    if checkpoint_data["git_sha"] != inference_sha:
        log.warning("WARNING: checkpoint's git sha is different from the current one. Usually nothing to worry about.")
        log.warning(f"Chekcpoint sha: {checkpoint_data['git_sha']}") 

    network = instantiate(model_config.network)
    mup.set_base_shapes(network, base=model_config.mup_base_shapes)
    sequence_adapter = instantiate(model_config.sequence_adapter)
    action_tokenizer = instantiate(model_config.action_tokenizer)

    state_dict = remove_prefix(checkpoint_data['state_dict'], 'network')
    network.load_state_dict(state_dict, strict=True)
    
    network.to(device)
    sequence_adapter.to(device)
    action_tokenizer.to(device)
    
    if set_net_to_eval:
        network.eval()
        sequence_adapter.eval()
        action_tokenizer.eval()
    
    return network, sequence_adapter, action_tokenizer

def load_data(pickle_path, dataset_config, dataloader_config):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)
    
    dataset = SceneBasedNuplanDataset(
        pickle_data = pickle_data, 
        **dataset_config
    )
    
    dataloader = DataLoader(
        dataset,
        worker_init_fn=worker_rnd_init,
        collate_fn=custom_collate,
        **dataloader_config
    )
    
    return dataloader

def save_data(output_dir, image_paths, context_end_index, data):
    """
    data is a tensor of shape [B, T, ...]
    """
    for b, sequence_image_paths in enumerate(image_paths):
        for t, image_path in enumerate(sequence_image_paths[context_end_index:]):
            output_path = (output_dir / image_path).with_suffix('.pkl')
            
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
            np.save(output_path, data[b, t].cpu(), allow_pickle=False)

def infer(inference_config: DictConfig) -> None:
    """Run inference given checkpoint and a datamodule config.

    Args:
        config: A DictConfig configuration composed by Hydra.
    """
    
    checkpoint_name = Path(inference_config.checkpoint_name)
    
    model_log_dir = Path(inference_config.model_log_dir)
    checkpoint_path = model_log_dir / 'checkpoints' / checkpoint_name
    assert checkpoint_path.exists(), str(checkpoint_path)
    
    # remove .ckpt suffix to get the name of the checkpoint
    base_output_dir = Path(inference_config.output_dir) 
    base_output_dir = (
        base_output_dir / 
        model_log_dir.name / 
        'inference' / 
        checkpoint_name.with_suffix('')
    )
    
    log.info("Initializing...")
    if not base_output_dir.exists():
        log.info(f'"{base_output_dir}" does not exists... creating it.')
        base_output_dir.mkdir(parents=True, exist_ok=True)    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    
    # Get current git sha of this codebase
    repo = git.Repo(search_parent_directories=True)
    inference_sha = repo.head.object.hexsha
    
    log.info(f"Loading the model config from logged data...")
    with open(model_log_dir / 'hparams.yaml', 'r') as file:
        training_logged_config = yaml.safe_load(file)
    training_logged_config = OmegaConf.create(training_logged_config)
    model_config = training_logged_config.model
    
    print_log_and_current_config(inference_config, training_logged_config)
    
    log.info(f"Instantiating the model...")
    network, sequence_adapter, action_tokenizer = load_model(
        checkpoint_path, 
        model_config, 
        device, 
        inference_sha, 
        set_net_to_eval=True
    )
    
    log.info(f"Instantiating the dataset...")
    pickle_path = Path(inference_config.paths.pickle_path) / inference_config.pickle_name
    dataloader = load_data(pickle_path, inference_config.dataset_config, inference_config.dataloader_config)
    
    log.info(f"Instantiating samplers...")
    samplers = instantiate_samplers(inference_config.samplers)    
    
    thread_pool = ThreadPoolExecutor(max_workers=inference_config.max_threads)
    
    with torch.no_grad():
        threads = []
        for i, batch in enumerate(tqdm(dataloader, file=sys.stdout)):
            print()
            
            batch = move_data_to_device(batch, device=device)

            action_tokens = action_tokenizer(**batch)
            
            context_visual_tokens = batch['visual_tokens'][:, :batch['context_end_index']]
            
            context_action_tokens = action_tokens[:, :batch['context_end_index']]
            future_action_tokens = action_tokens[:, batch['context_end_index']:]
            
            for s, sampler in enumerate(samplers): 
                for sampling_idx in range(inference_config.nb_samplings):
                    
                    output_dir = (
                        base_output_dir / f'{sampler}_sampling' / f'generation_{sampling_idx}'
                    )
                    logits_output_dir = output_dir / 'logits'
                    tokens_output_dir = output_dir / 'tokens'
                    
                    generated_data = autoregressive_image_sequence_generation(
                        network,
                        sampler, 
                        sequence_adapter,
                        context_visual_tokens,
                        context_action_tokens,
                        future_action_tokens,
                        temperature = 1.0,
                        return_logits = sampling_idx == 0
                    )

                    if sampling_idx == 0:
                        # save logits
                        # generated_data['visual_logits'] is a list of t tensors of shape [B,H,W,vocab_size]
                        visual_logits = rearrange(generated_data['visual_logits'], 'b t h w vocab_size -> b t vocab_size h w')
                        args = (logits_output_dir, batch['images_paths'], batch['context_end_index'], visual_logits)
                        thread = thread_pool.submit(save_data, *args)
                        threads.append(thread)
                        
                    # save generated tokens
                    # generated_data['visual_tokens'] is a list of t tensors of shape [B,H,W]
                    args = (tokens_output_dir, batch['images_paths'], batch['context_end_index'], generated_data['visual_tokens'])
                    thread = thread_pool.submit(save_data, *args)
                    threads.append(thread)
                    
                    if (s+1)*(i+1)*(sampling_idx+1) % inference_config.concurrent_loops == 0:
                        # Wait for all threads to complete, and check if any exception raised.
                        wait(threads)
                        for thread in threads:
                            if thread.exception():
                                print(thread.result())
                        threads = []
                        
    OmegaConf.update(inference_config, 'git_sha', inference_sha, force_add=True)    
    meta_config = DictConfig({
        'inference_config': inference_config,
        'training_config': training_logged_config
    })
    
    for s, sampler in enumerate(samplers): 
        metada_path = base_output_dir / f'{sampler}_sampling' / 'generated_tokens_metadata.yaml'
        OmegaConf.save(meta_config, metada_path)
    
    log.info("Processing complete!")
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(config: DictConfig) -> None:
    """Main entry point for inference.

    Args:
        config: DictConfig configuration composed by Hydra.
    """
    extras(config)
    
    infer(config)


if __name__ == "__main__":
    main()