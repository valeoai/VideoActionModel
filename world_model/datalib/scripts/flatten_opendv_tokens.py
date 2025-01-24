"""
Example usage.

python world_model/datalib/scripts/flatten_opendv_tokens.py \
--rootdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/tokens \
--outdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens
"""

import argparse
import os
from glob import glob


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


parser = argparse.ArgumentParser()
parser.add_argument("--rootdir", type=_path, required=True, help="Root directories to flatten.")
parser.add_argument("--outdir", type=_path, required=True, help="Flatten dir.")
args = parser.parse_args()


all_subdirectory = glob(os.path.join(args.rootdir, "**", "*"))
print(f"Found {len(all_subdirectory)} video files.")

os.makedirs(args.outdir, exist_ok=True)
for video_path in all_subdirectory:
    new_path = os.path.join(args.outdir, os.path.basename(video_path))
    if os.path.exists(new_path):
        continue
    os.symlink(video_path, new_path)
    print(f"Created symlink {video_path} -> {new_path}")
