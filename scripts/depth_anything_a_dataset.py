"""
Example usage:
python scripts/depth_anything_a_dataset.py --dataset_name cityscapes
python scripts/depth_anything_a_dataset.py --dataset_name kitti
python scripts/depth_anything_a_dataset.py --dataset_name kitti_video

python scripts/depth_anything_a_dataset.py --dataset_name cityscapes --compute_only_issues
"""

import argparse
import os
import sys
import time
from glob import glob
from queue import Queue
from threading import Thread
from typing import Any, Dict, List

import cv2
import git
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from vam.evaluation.datasets import CityscapesDataset, KITTIDataset
from vam.utils import expand_path, read_eval_config

try:
    print("Cloning Depth-Anything repository")
    git.Repo.clone_from("https://github.com/LiheYoung/Depth-Anything.git", "Depth-Anything")
    print("Cloning finished")
    os.system("rm -rf Depth-Anything/.git")  # remove the .git folder to avoid vscode from indexing it
except git.exc.GitCommandError:
    pass

sys.path.append("Depth-Anything")

if not os.path.exists("Depth-Anything/checkpoints/depth_anything_vitl14.pth"):
    print("Downloading Depth-Anything checkpoint")
    os.makedirs("Depth-Anything/checkpoints", exist_ok=True)
    os.system(
        "wget https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth "
        "-O Depth-Anything/checkpoints/depth_anything_vitl14.pth"
    )

if not os.path.exists("~/.cache/torch/hub/facebookresearch_dinov2_main"):
    _ = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")

# make a symlink between ~/.cache/torch/hub and torchhub
if not os.path.exists("torchhub"):
    os.makedirs("torchhub", exist_ok=True)
    # there is a typo in the depth anything code, so we need to create a symlink
    os.system("ln -s ~/.cache/torch/hub/facebookresearch_dinov2_main torchhub/")


from depth_anything.dpt import DepthAnything  # noqa: E402 # type: ignore
from depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize  # noqa: E402  # type: ignore


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="cityscapes")
parser.add_argument("--config", type=read_eval_config, default=read_eval_config("configs/paths/eval_paths_jeanzay.yaml"))
parser.add_argument("--outdir", type=expand_path, default="")
parser.add_argument("--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl"])
parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="do not apply colorful palette")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--num_save_workers", type=int, default=8)
parser.add_argument("--max_queue_size", type=int, default=1000)
parser.add_argument("--compute_only_issues", action="store_true", help="only compute issues")
args = parser.parse_args()


DATASET_CONFIG = {
    "cityscapes": {
        "root": args.config["cityscapes"]["root"],
        "target_size": (288, 512),
    },
    "kitti": {
        "root": args.config["kitti"]["root"],
        "window_size": 1,
        "frame_stride": 1,
        "target_size": (288, 512),
    },
    "kitti_video": {
        "root": args.config["kitti_video"]["root"],
        "window_size": 8,
        "frame_stride": 5,
        "target_size": (288, 512),
    },
}

rank = 0
args.outdir = os.path.join(args.outdir, args.dataset_name + "_depth")
os.makedirs(args.outdir, exist_ok=True)

h, w = DATASET_CONFIG[args.dataset_name]["target_size"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
depth_anything = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder)).to(DEVICE).eval()

total_params = sum(param.numel() for param in depth_anything.parameters())
print("Total parameters: {:.2f}M".format(total_params / 1e6))


def load_image(pth: str) -> Dict[str, Any]:
    image = cv2.imread(pth)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    return {"image": image}


load_and_transform = Compose(
    [
        load_image,
        Resize(
            width=(w // 14) * 14,
            height=(h // 14) * 14,
            resize_target=False,
            keep_aspect_ratio=False,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        lambda x: torch.from_numpy(x["image"]),
    ]
)


if args.dataset_name == "cityscapes":
    dts_fn = CityscapesDataset
elif args.dataset_name.startswith("kitti"):
    dts_fn = KITTIDataset
train_dataset = dts_fn(**DATASET_CONFIG[args.dataset_name], split="train")
val_dataset = dts_fn(**DATASET_CONFIG[args.dataset_name], split="val")

train_dataset.load_and_transform = load_and_transform
val_dataset.load_and_transform = load_and_transform

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)


@torch.no_grad()
def process_batch_images(image: Tensor) -> np.ndarray:
    depth = depth_anything(image)

    depth = F.interpolate(depth.unsqueeze(1), (h, w), mode="bilinear", align_corners=False).squeeze(1)
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)
    return depth


def save_result(queue: Queue, rank: int) -> None:
    total_saved = 0
    start_time = time.time()
    while True:
        item = queue.get()
        if item is None:
            break

        output_path, depth = item
        cv2.imwrite(output_path, depth)

        total_saved += 1
        if total_saved % 1000 == 0:
            elapsed_time = time.time() - start_time
            avg_save_time = elapsed_time / total_saved
            queue_size = queue.qsize()
            print(
                f"Rank {rank}: Saved {total_saved} depth maps. "
                f"Average save time: {avg_save_time:.4f} seconds per file. "
                f"Current queue size: {queue_size}"
            )

        queue.task_done()

    print(f"Save thread finished. Total saved: {total_saved}")


def find_issues() -> List[str]:
    all_paths = glob(os.path.join(args.outdir, "**/*.png"), recursive=True)
    issues = []
    for path in tqdm(all_paths, desc="Checking issues"):
        try:
            img = Image.open(path)
            _ = TF.to_image(img)
        except Exception:
            issues.append(path)
    return issues


if args.compute_only_issues:
    issues = find_issues()
    issues = [os.path.basename(x).split("_")[0] for x in issues]
    print(f"Found {len(issues)} issues")

    if len(issues) == 0:
        print("No issues found. Exiting.")
        sys.exit(0)


if not args.compute_only_issues:
    nb_save_workers = args.num_save_workers
    save_queue = Queue(maxsize=args.max_queue_size)
    save_threads = []
    for _ in range(nb_save_workers):
        thread = Thread(target=save_result, args=(save_queue, rank))
        thread.start()
        save_threads.append(thread)
    print("Save threads started")


for split, loader in [("train", train_loader), ("val", val_loader)]:
    local_output_dir = os.path.join(args.outdir, split)
    os.makedirs(local_output_dir, exist_ok=True)

    for batch in tqdm(loader, desc=f"[{split}]"):
        if args.compute_only_issues:
            if batch["image"].ndim == 5:
                all_path_batch = [pth for pths in batch["image_path"] for pth in pths]
            else:
                all_path_batch = batch["image_path"]
            if not any([y in x for y in issues for x in all_path_batch]):  # noqa: C419
                continue

        image = batch["image"].to(DEVICE)

        if image.ndim == 5:  # this means that the input is a video
            image = rearrange(image, "b t c h w -> (b t) c h w")
            # flatten the List[List[str]] to List[str]
            batch["image_path"] = [pth for pths in batch["image_path"] for pth in pths]

        depth = process_batch_images(image)

        for j, file_path in enumerate(batch["image_path"]):
            if args.grayscale:
                _d = np.repeat(depth[j][..., np.newaxis], 3, axis=-1)
            else:
                _d = cv2.applyColorMap(depth[j], cv2.COLORMAP_INFERNO)

            filename = loader.dataset.get_unique_identifier_from_path(file_path)
            filename = filename[: filename.rfind(".")] + "_depth.png"
            output_path = os.path.join(local_output_dir, filename)
            if args.compute_only_issues:
                cv2.imwrite(output_path, _d)
            else:
                save_queue.put((output_path, _d))


if not args.compute_only_issues:
    # Sending None to Threads is an exit signal
    for _ in range(nb_save_workers):
        save_queue.put(None)
    # blocks the main thread until the all thread has completed
    for thread in save_threads:
        thread.join()
    print("Save queue finished")

issues = find_issues()
if len(issues) > 0:
    print(f"Found {len(issues)} issues")
    print("Please relaunch the script with `--compute_only_issues` to solve the issues")
