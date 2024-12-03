import os
import json
import argparse
import logging
import sys
from typing import Optional

import pandas as pd
from colorlog import ColoredFormatter

from world_model.opendv import TokenCreator


def setup_logger(logdir: Optional[str] = None) -> None:
    logging.getLogger().handlers.clear()
    formatter = ColoredFormatter(
        "[%(cyan)s%(asctime)s%(reset)s]"
        "[%(light_blue)s%(name)s%(reset)s]"
        "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red",
        },
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    if logdir is not None:
        logfile = os.path.join(logdir, "log.txt")
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(file_handler)

    default_level = os.getenv("LOGGING_LEVEL", "INFO")

    logging.getLogger().setLevel(default_level)

    return logging.getLogger()


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


parser = argparse.ArgumentParser(description="OpenDV Token Creator")
parser.add_argument("--video_list", type=_path, help="Path to the video list (json)")
parser.add_argument("--metadata", type=_path, help="Path to the metadata file")
parser.add_argument("--outdir", type=_path, help="Output directory")
parser.add_argument("--tokenizer_jit_path", type=_path, help="Path to the tokenizer jit model")
parser.add_argument("--num_frames_threads", type=int, help="Number of threads for frame extraction")
parser.add_argument("--num_writer_threads", type=int, help="Number of threads for writing to disk")
parser.add_argument("--frames_queue_size", type=int, help="Size of the frames queue")
parser.add_argument("--writer_queue_size", type=int, help="Size of the writer queue")
parser.add_argument("--batch_size", type=int, help="Batch size for tokenization")
parser.add_argument("--target_frame_rate", type=int, help="Target frame rate for ffmpeg")
parser.add_argument("--target_width", type=int, help="Target width for resizing")
parser.add_argument("--target_height", type=int, help="Target height for resizing")
parser.add_argument("--keep_temp_frames", action="store_true", help="Keep temporary frames after tokenization")
args = parser.parse_args()
setup_logger(logdir='./')

if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    # multiprocessing with torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
elif "SLURM_PROCID" in os.environ:
    # multiprocessing with slurm
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
else:
    # single process
    rank = 0
    world_size = 1

# Load the video list
with open(args.video_list, "r") as f:
    video_list = json.load(f)

# Divide video_list into `world_size` parts
video_list = video_list[rank::world_size]

# Load the metadata
metadata = pd.read_csv(args.metadata, sep="\t")


# Create the dataset creator
creator = TokenCreator(
    video_list=video_list,
    metadata=metadata,
    outdir=args.outdir,
    rank=rank,
    tokenizer_jit_path=args.tokenizer_jit_path,
    num_frames_threads=args.num_frames_threads,
    num_writer_threads=args.num_writer_threads,
    frames_queue_size=args.frames_queue_size,
    writer_queue_size=args.writer_queue_size,
    batch_size=args.batch_size,
    target_frame_rate=args.target_frame_rate,
    target_width=args.target_width,
    target_height=args.target_height,
    remove_temp_frames=not args.keep_temp_frames,
)

# Create the dataset
dataset_info = creator.create_opendv_dataset()

for key, value in dataset_info.items():
    print(f"{key}: {value}")
