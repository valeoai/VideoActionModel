import random
from typing import List, Optional

from torch.utils.data import ConcatDataset, Dataset, Subset

from world_model.opendv.ego_trajectory_dataset import EgoTrajectoryDataset
from world_model.opendv.random_tokenized_sequence_opendv import RandomTokenizedSequenceOpenDVDataset


def _subsample_dataset(dataset: Dataset, n_samples: int, seed: int = 0) -> Subset:
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:n_samples])


def _oversample_dataset(dataset: Dataset, n_samples: int, seed: int = 0) -> ConcatDataset:
    to_concat = [dataset] * (n_samples // len(dataset))
    if n_samples % len(dataset) != 0:
        to_concat.append(_subsample_dataset(dataset, n_samples % len(dataset), seed))
    return ConcatDataset(to_concat)


def mix_datasets(
    datasets: List[Dataset],
    ratios: List[float],
    total_number_of_samples: int,
    seed: int = 0,
) -> ConcatDataset:
    new_dataset_size = [int(r * total_number_of_samples) for r in ratios]

    final_datasets = []
    for target_size, dts in zip(new_dataset_size, datasets):
        if target_size == len(dts):
            final_datasets.append(dts)
        elif target_size > len(dts):
            final_datasets.append(_oversample_dataset(dts, target_size, seed))
        else:
            final_datasets.append(_subsample_dataset(dts, target_size, seed))

    return ConcatDataset(final_datasets)


def all_token_datasets(
    opendv_data_root_dir: str,
    opendv_video_list: List[str],
    nuplan_pickle_data: List[dict],
    nuplan_tokens_rootdir: str,
    nuscenes_pickle_data: List[dict],
    nuscenes_tokens_rootdir: str,
    sequence_length: int = 8,
    ratios: Optional[List[float]] = None,
    total_number_of_samples: Optional[int] = None,
    seed: int = 0,
) -> ConcatDataset:
    opendv_dataset = RandomTokenizedSequenceOpenDVDataset(
        data_root_dir=opendv_data_root_dir,
        video_list=opendv_video_list,
        sequence_length=sequence_length,
        subsampling_factor=5,
    )

    nuscenes_dataset = EgoTrajectoryDataset(
        pickle_data=nuscenes_pickle_data,
        tokens_rootdir=nuscenes_tokens_rootdir,
        tokens_only=True,
        sequence_length=sequence_length,
    )

    nuplan_dataset = EgoTrajectoryDataset(
        pickle_data=nuplan_pickle_data,
        tokens_rootdir=nuplan_tokens_rootdir,
        tokens_only=True,
        sequence_length=sequence_length,
        camera="CAM_F0",
        subsampling_factor=5,
    )

    if ratios is None:
        return ConcatDataset([opendv_dataset, nuscenes_dataset, nuplan_dataset])

    return mix_datasets(
        datasets=[opendv_dataset, nuscenes_dataset, nuplan_dataset],
        ratios=ratios,
        total_number_of_samples=total_number_of_samples,
        seed=seed,
    )
