import argparse
import os
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vam.datalib import CropAndResizeTransform, EgoTrajectoryDataset, torch_image_to_plot
from vam.evaluation.datasets import KITTIDataset
from vam.utils import boolean_flag, create_mp4_from_folder, expand_path, read_eval_config, torch_dtype
from vam.video_pretraining import MupGPT2, load_pretrained_gpt

ImageType = Tensor | np.ndarray | List[Tensor] | List[np.ndarray]
Config = Dict[str, Any]

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
_DISABLE_TQDM = os.environ.get("DISABLE_TQDM", False)

plt.style.use("default")
plt.rcParams.update(
    {
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)


def get_nuscenes(config: Config, context_length: int) -> EgoTrajectoryDataset:
    with open(expand_path(config["nuscenes"]["pickle"]), "rb") as f:
        pickle_data = pickle.load(f)

    transform = CropAndResizeTransform(resize_factor=3.125, trop_crop_size=0)

    return EgoTrajectoryDataset(
        pickle_data=pickle_data,
        images_rootdir=expand_path(config["nuscenes"]["images_rootdir"]),
        sequence_length=context_length,
        images_transform=transform,
    )


def get_kitti(config: Config, context_length: int) -> KITTIDataset:
    return KITTIDataset(
        root=config["kitti"]["root"],
        split="val",
        window_size=context_length,
        frame_stride=5,
        eval_on_last_frame=True,
    )


def save_images(images: ImageType, savedir: str) -> None:
    for idx, img in enumerate(images):
        if isinstance(img, Tensor):
            img = img.cpu().numpy()
        img = Image.fromarray(img)
        img.save(os.path.join(savedir, f"{idx:04d}.png"))
        img.close()


def plot_images(
    images: ImageType,
    save_path: Optional[str] = None,
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
) -> None:
    num_rows = len(images) if isinstance(images, list) else 1
    num_cols = max(len(img) for img in images) if isinstance(images, list) else len(images)

    fig = plt.figure(figsize=(20, 5))
    axes = fig.subplots(num_rows, num_cols)

    idx = 0
    for row in range(num_rows):
        if isinstance(images, list):
            idx = 0
        for col in range(num_cols):
            if idx >= (len(images[row]) if isinstance(images, list) else len(images)):
                break
            if isinstance(images, list):
                axes[row][col].imshow(images[row][idx])
            else:
                axes[row][col].imshow(images[idx])
            axes[row][col].axis("off")
            idx += 1
            # plot_idx += 1
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    # Ensure tight layout
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")


def handle_output(context_frames: np.ndarray, generated_frames: np.ndarray, window_idx: int, outdir: str) -> None:
    os.makedirs(window_outdir := os.path.join(outdir, window_idx), exist_ok=True)
    plot_images([context_frames, generated_frames], save_path=os.path.join(window_outdir, "plot.png"))
    os.makedirs(context_outdir := os.path.join(window_outdir, "context"), exist_ok=True)
    os.makedirs(generated_outdir := os.path.join(window_outdir, "generated"), exist_ok=True)
    save_images(context_frames, context_outdir)
    save_images(generated_frames, generated_outdir)
    create_mp4_from_folder(context_outdir, os.path.join(window_outdir, "context.mp4"))
    create_mp4_from_folder(generated_outdir, os.path.join(window_outdir, "generated.mp4"))
    print(f"Saved results for window {window_idx} to {window_outdir}")
    return


@torch.no_grad()
def get_future_frames(
    *,
    args: argparse.Namespace,
    outdir: str,
    gpt: MupGPT2,
    loader: DataLoader,
    tokenizer: Optional[torch.jit.ScriptModule] = None,
    detokenizer: torch.jit.ScriptModule,
    temperature: float = 1.0,
    topk_sampler: int = 1,
    world_size: int = 1,
) -> None:
    with ThreadPoolExecutor(max_workers=10) as plot_executor:
        num_samples = 0
        total_bs = world_size * args.per_proc_batch_size
        total = (
            len(loader)
            if args.generate_x == float("inf")
            else (args.generate_x // total_bs) + int((args.generate_x % total_bs) > 0)
        )
        for batch in tqdm(loader, total=total, disable=_DISABLE_TQDM):
            if num_samples >= args.generate_x:
                break
            num_samples += len(batch["window_idx"]) * world_size

            with torch.amp.autocast(args.device, dtype=args.dtype):
                if tokenizer is not None:
                    tokens = tokenizer(rearrange(batch["image"].to(args.device, non_blocking=True), "b t ... -> (b t) ..."))
                    tokens = rearrange(tokens, "(b t) ... -> b t ...", t=args.context_length)
                else:
                    tokens = batch["visual_tokens"].to(args.device, non_blocking=True)

                generated_tokens = gpt.forward_inference(
                    number_of_future_frames=args.prediction_length,
                    burnin_visual_tokens=tokens,
                    topk_sampler=topk_sampler,
                    temperature=temperature,
                    use_kv_cache=True,
                )

                generated_images = detokenizer(rearrange(generated_tokens, "b t ... -> (b t) ..."))
                generated_images = torch_image_to_plot(generated_images, to_numpy=False)
                generated_images = rearrange(generated_images, "(b t) ... -> b t ...", t=args.prediction_length).cpu().numpy()

            _images = rearrange(batch["image"], "b t ... -> (b t) ...")
            _images = torch_image_to_plot(_images, to_numpy=False)
            _images = rearrange(_images, "(b t) ... -> b t ...", t=args.context_length).cpu().numpy()
            plot_futures = []
            for idx, window_idx in enumerate(batch["window_idx"].tolist()):
                future = plot_executor.submit(
                    # plot_images, [_images[idx], generated_images[idx]], save_path=os.path.join(outdir, f"{window_idx}.png")
                    handle_output,
                    _images[idx],
                    generated_images[idx],
                    window_idx,
                    outdir,
                )
                plot_futures.append(future)

        for future in as_completed(plot_futures):
            future.result()


if __name__ == "__main__":
    """
    Example usage:

    python scripts/video_qualitative_results.py \
        --config configs/paths/eval_paths_local.yaml \
        --outdir ~/iveco/scratch_iveco/VAM_JZGC4/video_qual_results/vavim_l \
        --gpt_checkpoint_path ~/iveco/scratch_iveco/VAM_JZGC4/checkpoints/Finetune/width_2048_pretrained_139k_total_155k.pt \
        --dtype bf16 \
        --generate_x 30 \
        --per_proc_batch_size 8
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=expand_path, required=True)
    parser.add_argument("--config", type=read_eval_config, default=read_eval_config("configs/paths/eval_paths_jeanzay.yaml"))

    parser.add_argument("--tokenizer_only", type=boolean_flag, default=False)
    parser.add_argument("--gpt_checkpoint_path", type=expand_path, default=None)

    parser.add_argument("--context_length", type=int, default=4)
    parser.add_argument("--prediction_length", type=int, default=4)

    parser.add_argument("--generate_x", type=int, default=float("inf"))

    parser.add_argument("--per_proc_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16)
    args = parser.parse_args()

    all_datasets = {
        "nuscenes": get_nuscenes(args.config, context_length=args.context_length),
        # "kitti": get_kitti(args.config, context_length=args.context_length + args.prediction_length),
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

    tokenizer = torch.jit.load(expand_path(args.config["tokenizer_jit_path"])).to("cuda")
    detokenizer = torch.jit.load(expand_path(args.config["detokenizer_jit_path"])).to("cuda")
    gpt = (
        None
        if args.tokenizer_only
        else load_pretrained_gpt(args.gpt_checkpoint_path, tempdir=os.environ.get("JOBSCRATCH", "/tmp"))
    )

    for name, ds in all_datasets.items():
        outdir = os.path.join(args.outdir, name)
        os.makedirs(outdir, exist_ok=True)

        sampler = None if world_size == 1 else DistributedSampler(ds, shuffle=args.generate_x < float("inf"))
        loader = DataLoader(
            ds,
            batch_size=args.per_proc_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=(not world_size > 1) and args.generate_x < float("inf"),
        )

        _ = get_future_frames(
            args=args,
            outdir=outdir,
            gpt=gpt,
            loader=loader,
            tokenizer=tokenizer,
            detokenizer=detokenizer,
        )
