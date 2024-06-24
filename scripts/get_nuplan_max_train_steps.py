import pickle
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from world_model.dataloader.components.random_tokenized_sequence_nuscenes import RandomTokenizedSequenceNuscenesDataset
from world_model.dataloader.tokenized_sequence_nuscenes import custom_collate


if __name__ == "__main__":
    pickle_data_path = '/lus/work/CT10/cin4181/SHARED/datasets_processed/nuplan_pickle/filtered_trainval_data.pkl'

    with open(pickle_data_path, 'rb') as rb:
        pickle_data = pickle.load(rb)

    lens = dict()

    for seq_len in tqdm(range(1,40)):
        train_nuplan_data = RandomTokenizedSequenceNuscenesDataset(
            quantized_nuscenes_root_dir = '/lus/work/CT10/cin4181/SHARED/datasets_processed/VQGAN_ImageNet_f16_1024',
            nuscenes_pickle_data = pickle_data['train'],
            transform = None,
            sequence_length = seq_len,
            nuscenes_root_dir = None,
            load_image = False
        )

        lens[seq_len] = len(train_nuplan_data)


    print(json.dumps({**(lens)}, indent=4))

