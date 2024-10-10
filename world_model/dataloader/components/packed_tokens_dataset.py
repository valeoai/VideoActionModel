'''
Dataloader for OpenDV tokens reading the data from the shards in a strictly sequential way.
The data must arrtive pre-shuffled and pre-scheduled by the tools in `assemble_shards/`.
'''
from collections import deque
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from world_model.dataloader.components.packed_tokens import HEADER_SIZE, decode_header, decode_record, get_record_format

class TokensRawDataset(IterableDataset):
    # pylint: disable=abstract-method
    '''
    Dataset for raw packed tokens files. This reads the scheduling information, one token at a time, without rebuilding
    the windows. See `packed_tokens.py` for more information on record format. Objects in this class are pickable,
    retaining the current position in the file.

    CAVEAT: A single iterator per instance is allowed, as the state of the iterator is stored in the object itself.

    Args:
        tokens_path (str): path to the packed tokens file.
    '''
    def __init__(self, tokens_path):
        self.tokens_path = tokens_path
        self.tokens_file = open(tokens_path, 'rb')
        self.header_bytes = self.tokens_file.read(HEADER_SIZE)
        self.header = decode_header(self.header_bytes)
        self.records_n = self.header['records_n']
        self.record_format, self.record_len = get_record_format(height=self.header['height'],
                                                                width=self.header['width'],
                                                                video_id_len=self.header['video_id_len'])
        self.record_i = 0

    def __del__(self):
        self.tokens_file.close()

    def __len__(self):
        return self.header['records_n']

    def __iter__(self):
        return self

    def __next__(self):
        if self.record_i == self.records_n:
            raise StopIteration
        self.record_i += 1
        record_bytes = self.tokens_file.read(self.record_len)
        return decode_record(record_bytes, record_fmt=self.record_format, header=self.header)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['tokens_file'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokens_file = open(self.tokens_path, 'rb')
        self.tokens_file.seek(HEADER_SIZE + self.record_i * self.record_len)


class TokensWindowedDataset(IterableDataset):
    # pylint: disable=abstract-method
    '''
    Dataset for packed tokens files. Reads a packed tokens file and returns the rebuilded windows.
    The packed file comes already perfectly scheduled, so this class just builds a circular buffer for each scheduler,
    and emits one window whenever the buffer is full. Objects in this class are pickable, retainin full state.

    CAVEAT: A single iterator per instance is allowed, as the state of the iterator is stored in the object itself.

    Args:
        tokens_path (str): path to the packed tokens file.
    '''
    def __init__(self, tokens_path):
        self.raw_dataset = TokensRawDataset(tokens_path)
        self.header = self.raw_dataset.header
        self.streams = self.header['schedulers_n']
        self.window_size = self.header['window_size']
        self.windows_n = self.header['windows_n']
        self.schedulers = tuple(deque([], maxlen=self.window_size) for _ in range(self.streams))
        self.frame_interval = self.header['frame_interval']

    def __len__(self):
        return self.windows_n

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            record = next(self.raw_dataset)
            scheduler = self.schedulers[record['scheduler']]
            if record['restart']:
                scheduler.clear()
            else:
                next_number = scheduler[-1]['frame_number'] + self.frame_interval
                if record['frame_number'] != next_number:
                    raise ValueError(f'Unexpected frame_number on record {self.raw_dataset.record_i}: '
                                     f'expected {next_number}, got {record["frame_number"]}')
            scheduler.append(record)
            if len(scheduler) == self.window_size:
                visual_tokens = np.stack([record['tokens'] for record in scheduler], axis=0)
                visual_tokens = torch.tensor(visual_tokens)
                return {'visual_tokens': visual_tokens}


class TokensWindowedMultishards(IterableDataset):
    # pylint: disable=abstract-method
    '''
    Creates a `TokensWindowedDataset` dataset from multiple shards, and let each worker of the DataLoader select a
    different shard. Objects in this class are pickable, retaining full state.
    '''
    def __init__(self, shard_paths, shard_index=None):
        self.shards = shard_paths
        self._shard_index = shard_index
        self.shards_n = len(shard_paths)
        self.datasets = [TokensWindowedDataset(shard_path) for shard_path in shard_paths]
        self.windows_n = sum(len(dataset) for dataset in self.datasets)
        self.dataset = None

    @property
    def shard_index(self):
        return self._shard_index

    @shard_index.setter
    def shard_index(self, value):
        if value < 0 or value >= self.shards_n:
            raise ValueError(f'Shard index ({value}) must be in the range [0, {self.shards_n})')
        self._shard_index = value
        self.dataset = self.datasets[self.shard_index]

    def __iter__(self):
        if self.dataset is None:
            raise ValueError('Shard index must be set before using the multi-shard dataset')
        return self

    def get_length(self):
        # CAVEAT: This is not implemented as __len__ to signal to the DataLoader to not use it.
        return self.windows_n

    def __next__(self):
        if self.dataset is None:
            raise ValueError('Shard index must be set before using the multi-shard dataset')
        return next(self.dataset)


def tokens_windowed_loader_worker_init_fn(worker_id):
    '''
    Ensures that each worker reads a different shard from the dataset.
    '''
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    assert worker_id >= 0 and worker_id < dataset.shards_n, \
           f'Worker ID ({worker_id}) must be in the range [0, {dataset.shards_n})'
    assert worker_info.num_workers == dataset.shards_n, \
           f'Number of workers ({worker_info.num_workers}) must match the number of shards ({dataset.shards_n})'
    dataset.shard_index = worker_id


class TokensWindowedLoader(DataLoader):
    '''
    DataLoader for `TokensWindowedMultishards`. Each worker reads an independent scheduled dataset.
    '''
    def __init__(self, shard_paths, **kwargs):
        num_workers = kwargs.setdefault('num_workers', len(shard_paths))
        if num_workers != len(shard_paths):
            raise ValueError(f'Number of workers ({num_workers}) must match the number of shards ({len(shard_paths)})')
        if 'shuffle' in kwargs:
            warnings.warn('TokensWindowedLoader: data in shards is pre-shuffled, so setting `shuffle` has no effect')
        kwargs.setdefault('worker_init_fn', tokens_windowed_loader_worker_init_fn)
        kwargs.setdefault('pin_memory', True)
        kwargs['shuffle'] = False
        super().__init__(dataset=TokensWindowedMultishards(shard_paths), **kwargs)