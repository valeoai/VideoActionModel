import argparse
import json
import os
import pickle
import subprocess

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vam.datalib import CropAndResizeTransform, EgoTrajectoryDataset
from vam.evaluation import MultiInceptionMetrics
from vam.evaluation.datasets import KITTIDataset
from vam.utils import boolean_flag, expand_path, torch_dtype
from vam.video_pretraining import MupGPT2, load_pretrained_gpt

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
_DISABLE_TQDM = os.environ.get("DISABLE_TQDM", False)


def get_kitti(context_length: int) -> KITTIDataset:
    return KITTIDataset(
        root="/datasets_local/KITTI_STEP",
        # root="$ycy_ALL_CCFRSCRATCH/KITTI_STEP",
        split="val",
        window_size=context_length,
        frame_stride=5,
        eval_on_last_frame=True,
    )


def get_nuscenes(context_length: int) -> EgoTrajectoryDataset:
    # with open(expand_path("$ycy_ALL_CCFRWORK/cleaned_trajectory_pickle/nuscenes_val_data_cleaned.pkl"), "rb") as f:
    with open(expand_path("pickles/nuscenes_val_data_cleaned.pkl"), "rb") as f:
        pickle_data = pickle.load(f)

    transform = CropAndResizeTransform(resize_factor=3.125, trop_crop_size=0)

    return EgoTrajectoryDataset(
        pickle_data=pickle_data,
        # images_rootdir=expand_path("$ycy_ALL_CCFRSCRATCH/nuscenes_v2"),
        images_rootdir=expand_path("/datasets_local/nuscenes"),
        sequence_length=context_length,
        images_transform=transform,
    )


@torch.no_grad()
def evaluate_a_dataset(
    args: argparse.Namespace,
    dataset: Dataset,
    tokenizer: nn.Module,
    detokenizer: nn.Module,
    gpt: MupGPT2,
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    sampler = None if not is_distributed else DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Quality metrics
    fid_evaluator = {}
    for k in args.fid_at:
        fid_evaluator[k] = MultiInceptionMetrics("cuda", model="dinov2")

    loader = tqdm(loader, disable=_DISABLE_TQDM or rank != 0)
    num_samples = 0
    for i, batch in enumerate(loader):
        if num_samples >= args.stop_after_x:
            break
        if _DISABLE_TQDM and rank == 0 and i % 100 == 0:
            print(f"Processing batch {i + 1}/{len(loader)}")

        images = batch["image"].to(device, non_blocking=True)
        batch_size = images.size(0)
        with torch.amp.autocast("cuda", dtype=dtype):
            # If tokenizer_only, we only encode the future frames
            loader.set_description("Creating the tokens...")
            to_tokenize = images[:, args.context_length :] if args.tokenizer_only else images[:, : args.context_length]
            time = to_tokenize.size(1)
            to_tokenize = rearrange(to_tokenize, "b t ... -> (b t) ...")
            visual_tokens = tokenizer(to_tokenize)

            if not args.tokenizer_only:
                loader.set_description("Generating future frames...")
                visual_tokens = rearrange(visual_tokens, "(b t) ... -> b t ...", t=time)
                visual_tokens = gpt.forward_inference(
                    number_of_future_frames=args.prediction_length,
                    burnin_visual_tokens=visual_tokens,
                    temperature=args.temperature,
                    topk_sampler=args.topk_sampler,
                )
                visual_tokens = rearrange(visual_tokens, "b t ... -> (b t) ...")

            loader.set_description("Detokenizing the frames...")
            future_generated_frames = detokenizer(visual_tokens)
            future_generated_frames = rearrange(future_generated_frames, "(b t) ... -> b t ...", t=args.prediction_length)

        future_generated_frames = future_generated_frames.float()

        for k, fid in fid_evaluator.items():
            fid.update(rearrange(images[:, : args.context_length], "b t ... -> (b t) ..."), image_type="real")
            fid.update(future_generated_frames[:, k - 1], image_type="fake")

        num_samples += batch_size * world_size

    # aggregate metrics
    _ = None if not is_distributed else dist.barrier()

    all_metrics = {}
    for k, fid in fid_evaluator.items():
        all_metrics[f"FID@{k}"] = fid.compute()["FID"]

    if rank == 0:
        for key, value in all_metrics.items():
            print(f"{key}: {value:.4f}")

    return all_metrics


if __name__ == "__main__":
    """
    module purge
    module load arch/h100
    module load pytorch-gpu/py3/2.4.0
    export PYTHONUSERBASE=$WORK/python_envs/world_model

    srun -A ycy@h100 -C h100 --pty \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread \
    --qos=qos_gpu_h100-dev --time=00:40:00 bash

    python scripts/quality_evaluation.py \
        --outfile ./tmp/reconstruction_quality_eval.json \
        --gpt_checkpoint_path xxx \
        --tokenizer_jit_path $ycy_ALL_CCFRWORK/llamagen_jit_models/VQ_ds16_16384_llamagen_encoder.jit \
        --detokenizer_jit_path $ycy_ALL_CCFRWORK/llamagen_jit_models/VQ_ds16_16384_llamagen_decoder.jit \
        --dtype bf16 \
        --stop_after_x 64 \
        --per_proc_batch_size 32

    # 768 --> Batch size: 32 / time: ~4h
    # 1024 --> Batch size: 32 / time: ~6h
    # 2048 --> Batch size: 32 / time: ~10h

    python scripts/quality_evaluation.py \
        --outfile ./tmp/reconstruction_quality_eval_llamagen.json \
        --gpt_checkpoint_path xxx \
        --tokenizer_jit_path ~/iveco/scratch_iveco/llamagen_jit_models/VQ_ds16_16384_llamagen_encoder.jit \
        --detokenizer_jit_path ~/iveco/scratch_iveco/llamagen_jit_models/VQ_ds16_16384_llamagen_decoder.jit \
        --dtype bf16 \
        --tokenizer_only True \
        --per_proc_batch_size 32
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", type=expand_path, required=True)

    parser.add_argument("--tokenizer_only", type=boolean_flag, default=False)
    parser.add_argument("--tokenizer_jit_path", type=expand_path, required=True)
    parser.add_argument("--detokenizer_jit_path", type=expand_path, required=True)
    parser.add_argument("--gpt_checkpoint_path", type=expand_path, default=None)

    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--prediction_length", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topk_sampler", type=int, default=1)
    parser.add_argument("--number_of_futures", type=int, default=1)
    parser.add_argument("--deterministic", type=boolean_flag, default=True)

    parser.add_argument("--fid_at", type=int, default=None, nargs="+")

    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--per_proc_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--stop_after_x", type=int, default=float("inf"))

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16)
    args = parser.parse_args()

    if args.fid_at is None:
        args.fid_at = range(1, args.prediction_length + 1)

    if args.number_of_futures > 1 and args.deterministic:
        raise ValueError("Cannot use deterministic mode with multiple futures")

    if not args.deterministic and args.topk_sampler <= 1:
        raise ValueError("Topk sampler must be greater than 1 for stochastic sampling")

    all_datasets = {
        "nuscenes": get_nuscenes(context_length=args.context_length + args.prediction_length),
        "kitti": get_kitti(context_length=args.context_length + args.prediction_length),
    }

    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    rank = 0
    if world_size > 1:
        dist_url = "env://"
        dist_backend = "nccl"
        rank = int(os.environ["SLURM_PROCID"])
        node_list = os.environ["SLURM_NODELIST"]
        is_distributed = world_size > 1
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput("scontrol show hostname {} | head -n1".format(node_list))
        local_rank = rank % num_gpus
        os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = addr
        print(f"| distributed init (rank {rank}): {dist_url}")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank)

    tokenizer = torch.jit.load(args.tokenizer_jit_path).to("cuda")
    detokenizer = torch.jit.load(args.detokenizer_jit_path).to("cuda")
    gpt = (
        None
        if args.tokenizer_only
        else load_pretrained_gpt(args.gpt_checkpoint_path, tempdir=os.environ.get("JOBSCRATCH", "/tmp"))
    )

    metrics = {}
    for name, dts in all_datasets.items():
        _m = evaluate_a_dataset(
            args,
            dts,
            tokenizer,
            detokenizer,
            gpt,
            rank,
            world_size,
            is_distributed=world_size > 1,
            device="cuda",
            dtype=args.dtype,
        )
        metrics[name] = _m
        print(metrics)

        metrics["gpt_checkpoint_path"] = args.gpt_checkpoint_path
        metrics["tokenizer_jit_path"] = args.tokenizer_jit_path
        metrics["detokenizer_jit_path"] = args.detokenizer_jit_path

        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
        with open(args.outfile, "w") as f:
            json.dump(metrics, f, indent=4)
