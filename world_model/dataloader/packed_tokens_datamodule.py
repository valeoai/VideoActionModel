import os
import pickle
import re

import lightning as L

from world_model.dataloader.components.packed_tokens_dataset import TokensWindowedLoader


def get_gpu_shard_paths(shards_dir, shards_list, node_rank, gpu_rank, shard_ext='.shard',
                        filter_re='[^_]+_n{NODE_RANK}_g{GPU_RANK}-[0-9]+'):
    '''
    This is a helper function to filter a list of shards into a list of full paths to be used in a data module for a
    single GPU.

    Args:
        shards_dir (str): path to the directory containing the shards.
        shards_list (list of str): list of shard names.
        node_rank (int): the rank of the node in the multi-node setup.
        gpu_rank (int): the local rank of the GPU within the node.
        shard_ext (str): the extension for the shard file, usually '.shard' for tokens or '.tar' for frames
                         (default: '.shard').
        filter_re (str): the regular expression to filter the shards (default: '[^_]+_n{NODE_RANK}_g{GPU_RANK}-[0-9]+').

    Returns:
        list of str: the list of full paths to the shards.
    '''
    filter_re = filter_re.format(NODE_RANK=node_rank, GPU_RANK=gpu_rank)
    filter_re = re.compile(filter_re)
    return [os.path.join(shards_dir, shard + shard_ext) for shard in shards_list if filter_re.match(shard)]

def read_shard_list(shard_list_path):
    with open(shard_list_path, 'rt', encoding='utf-8') as f:
        shard_list = [r.strip() for r in f.readlines()]
        shard_list = [r for r in shard_list if r and not r.startswith('#')]
        return shard_list

class TokensWindowedDataModule(L.LightningDataModule):
    def __init__(self, shards_dir, shard_list_path, train_kwargs=None,  val_shard_list_path=None, val_kwargs=None,
                 test_kwargs=None, predict_kwargs=None, shard_ext='.shard'):
        super().__init__()
        
        self.shards_dir = shards_dir
        self.shard_list_path = shard_list_path
        self.val_shard_list_path = val_shard_list_path
        
        self.shard_ext = shard_ext
        
        self.train_kwargs = train_kwargs
        self.val_kwargs = val_kwargs
        self.test_kwargs = test_kwargs
        self.predict_kwargs = predict_kwargs
        
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.predict_loader = None

        # Read shard list
        self.shard_list = read_shard_list(self.shard_list_path)
            
        if self.val_shard_list_path:
            self.val_shard_list = read_shard_list(shard_list_path=self.val_shard_list_path)

    def setup(self, stage=None):
        # This method is called by PyTorch Lightning to setup the data for each GPU
        if self.trainer is None:
            raise ValueError("Trainer is not set. Make sure you're using this DataModule with a PyTorch Lightning Trainer.")

        node_rank = self.trainer.node_rank
        gpu_rank = self.trainer.global_rank % self.trainer.num_devices

        local_shard_list = get_gpu_shard_paths(self.shards_dir, self.shard_list, 
                                               node_rank=node_rank, gpu_rank=gpu_rank,
                                               shard_ext=self.shard_ext)

        if stage == 'fit' or stage is None:
            self.train_shard_paths = local_shard_list
            
            if self.val_shard_list:
                val_local_shard_list = get_gpu_shard_paths(self.shards_dir, self.val_shard_list, 
                                               node_rank=node_rank, gpu_rank=gpu_rank,
                                               shard_ext=self.shard_ext)
                self.val_shard_paths = val_local_shard_list

        if stage == 'test' or stage is None:
            self.test_shard_paths = local_shard_list if self.test_kwargs else None

        if stage == 'predict':
            self.predict_shard_paths = local_shard_list if self.predict_kwargs else None

    def train_dataloader(self):
        if not hasattr(self, 'train_shard_paths'):
            return None
        self.train_loader = TokensWindowedLoader(self.train_shard_paths, **self.train_kwargs)
        return self.train_loader

    def val_dataloader(self):
        if not hasattr(self, 'val_shard_paths'):
            return None
        self.val_loader = TokensWindowedLoader(self.val_shard_paths, **self.val_kwargs)
        return self.val_loader

    def test_dataloader(self):
        if not hasattr(self, 'test_shard_paths'):
            return None
        self.test_loader = TokensWindowedLoader(self.test_shard_paths, **self.test_kwargs)
        return self.test_loader

    def predict_dataloader(self):
        if not hasattr(self, 'predict_shard_paths'):
            return None
        self.predict_loader = TokensWindowedLoader(self.predict_shard_paths, **self.predict_kwargs)
        return self.predict_loader

    def state_dict(self):
        return {'data_module_pickled_state': pickle.dumps(self.__dict__)}

    def load_state_dict(self, state_dict):
        self.__dict__.update(pickle.loads(state_dict['data_module_pickled_state']))