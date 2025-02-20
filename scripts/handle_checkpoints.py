"""
Example usage:

1. Create tar files from checkpoints

python scripts/handle_checkpoints.py \
--mode create \
--checkpoint_dir ~/iveco/scratch_iveco/VAM_JZGC4/checkpoints \
--outdir weights/release \
--maxsize 1900MB

2. Extract tar files

python scripts/handle_checkpoints.py \
--mode extract \
--checkpoint_dir tmp/release/width_768_pretrained_139k \
--outdir tmp/release
"""

import argparse
import os
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from vam.utils import expand_path


def create_tar_file(checkpoint_path: str, outdir: str, maxsize: str) -> None:
    file_path = os.path.basename(checkpoint_path)
    file_dir = os.path.dirname(checkpoint_path)
    filename = os.path.basename(checkpoint_path).split(".")[0]
    outdir = os.path.join(outdir, filename)
    os.makedirs(outdir, exist_ok=True)
    part_name = os.path.join(outdir, f"{filename}_chunked.tar.gz.part_")
    subprocess.run(f"tar czf - -C {file_dir} {file_path} | split -b {maxsize} - {part_name}", shell=True)

    # check whether a unique tar file is created
    tar_files = glob(f"{part_name}*")
    if len(tar_files) > 1:
        return

    os.rename(tar_files[0], os.path.join(outdir, f"{filename}.tar.gz"))


def extract_tar_file(tar_dir: str, outdir: str) -> None:
    # Concatenate tar parts
    tmpdir = os.path.join(outdir, "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    part_name = os.path.join(tar_dir, "*.tar.gz.part_*")
    tmp_tar = os.path.join(tmpdir, "tempfile.tar.gz")
    subprocess.run(f"cat {part_name} > {tmp_tar}", shell=True)

    # Extract tar file
    weightdir = os.path.join(outdir, "weights")
    os.makedirs(weightdir, exist_ok=True)
    subprocess.run(f"tar xzf {tmp_tar} -C {weightdir}", shell=True)

    # Remove temporary files
    os.remove(tmp_tar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["create", "extract"])
    parser.add_argument("--checkpoint_dir", type=expand_path, required=True)
    parser.add_argument("--outdir", type=expand_path, required=True)
    parser.add_argument("--maxsize", type=str, default="1900MB")
    parser.add_argument("--extension", type=str, default="pt")
    parser.add_argument("--num_threads", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode == "create":
        if os.path.isdir(args.checkpoint_dir):
            checkpoint_paths = glob(os.path.join(args.checkpoint_dir, "**", f"*.{args.extension}"), recursive=True)
            print(f"Found {len(checkpoint_paths)} checkpoints")
        else:
            assert os.path.isfile(args.checkpoint_dir), "Invalid checkpoint path"
            checkpoint_paths = [args.checkpoint_dir]

        if args.num_threads <= 1:
            for checkpoint_path in tqdm(checkpoint_paths, desc="Creating tar files"):
                create_tar_file(checkpoint_path, args.outdir, args.maxsize)
        else:
            with ThreadPoolExecutor(max_workers=args.num_threads) as plot_executor:
                all_futures = []
                for checkpoint_path in checkpoint_paths:
                    future = plot_executor.submit(
                        create_tar_file, checkpoint_path, args.outdir, args.maxsize
                    )
                    all_futures.append(future)

                for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Creating tar files"):
                    future.result()

    elif args.mode == "extract":
        extract_tar_file(args.checkpoint_dir, args.outdir)
