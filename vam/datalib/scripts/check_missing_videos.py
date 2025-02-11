"""
Check OpenDV pre-processing.

Example:
python scripts/check_missing_videos.py \
--rootdir $fzh_ALL_CCFRSCRATCH/OpenDV_Youtube/videos \
--outdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/frames512
"""

import argparse
import os
from glob import glob
from typing import Dict, List


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


def get_videoids_from_files(rootdir: str) -> Dict[str, str]:
    """Get video ids from root directory."""
    _videos = glob(os.path.join(rootdir, "**", "*.mp4"), recursive=True)
    _videos += glob(os.path.join(rootdir, "**", "*.webm"), recursive=True)
    videos = {}
    for v in _videos:
        video_id = os.path.basename(v).split(".")[0]
        if video_id[0] == "-":
            video_id = "@" + video_id[1:]
        videos[video_id] = v
    return videos


def get_videoids_from_folder(rootdir: str) -> List[str]:
    """Get video ids from database."""
    videos = glob(os.path.join(rootdir, "**", "*"))
    videos = [os.path.basename(v) for v in videos]
    return videos


def get_missing_videos(rootdir: str, outdir: str) -> Dict[str, str]:
    """Get missing videos."""
    videos_in = get_videoids_from_files(rootdir)
    print(f"Number of videos in rootdir: {len(videos_in)}")
    videos_out = get_videoids_from_folder(outdir)
    print(f"Number of videos in outdir: {len(videos_out)}")
    missing = {k: v for k, v in videos_in.items() if k not in videos_out}
    print(f"Number of missing videos: {len(missing)}")
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=_path, required=True)
    parser.add_argument("--outdir", type=_path, required=True)
    args = parser.parse_args()

    missing = get_missing_videos(args.rootdir, args.outdir)
    # print(missing)
