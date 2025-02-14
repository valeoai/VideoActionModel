"""Small script to create .mp4 from NeuroNCAP logs."""

import argparse
import os
import subprocess
from glob import glob
from typing import List

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x  # noqa: E731

SCENARIOS = [
    "frontal",
    "side",
    "stationary",
]


def get_all_folders(rootdir: str) -> List[str]:
    """Get all folders for the scenarios, different scenes and runs."""
    all_folders = []
    for scenario in SCENARIOS:
        print(os.path.join(f"{rootdir}", f"{scenario}-*", "run_*"))
        all_folders.extend(glob(os.path.join(f"{rootdir}", f"{scenario}-*", "run_*")))
    return all_folders


def create_mp4_from_folder(folder: str, task: str, outdir: str) -> str:
    """Create a gif from a folder with image files."""
    name = f"{folder.split('/')[-2]}_{folder.split('/')[-1]}.mp4".replace("-", "_")
    name = os.path.join(outdir, name)
    if os.path.exists(name):
        print(f"Skipping {name} as it already exists.")
        return name
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-hide_banner",
        "-nostats",
        "-f",
        "image2",
        "-framerate",
        "2",
        "-pattern_type",
        "glob",
        "-i",
        "*.{jpg,png}",
        "-c:v",
        "libx264",  # Use H.264 codec
        "-pix_fmt",
        "yuv420p",  # Standard pixel format for compatibility
        "-preset",
        "medium",  # Encoding speed preset
        "-crf",
        "23",  # Quality setting (lower = better quality, 23 is default)
        name,
    ]
    subprocess.run(cmd, cwd=os.path.join(folder, task))
    return name


def main(rootdir: str, task: str) -> None:
    """Create .mp4 for all folders in the rootdir."""
    rootdir = os.path.expanduser(os.path.expandvars(rootdir))
    outdir = os.path.join(rootdir, f"mp4_{task}")
    os.makedirs(outdir, exist_ok=True)
    folders = get_all_folders(rootdir)
    print(f"Creating .mp4 for {len(folders)} folders.")
    for folder in tqdm(folders):
        create_mp4_from_folder(folder, task, outdir)


if __name__ == "__main__":
    from vam.utils import expand_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=expand_path)
    parser.add_argument("--task", type=str, choices=["CAM_FRONT", "FC_TRAJ", "COMBINED_OUTPUTS"], default="COMBINED_OUTPUTS")
    args = parser.parse_args()
    main(args.rootdir, args.task)
