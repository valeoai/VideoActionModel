import argparse
import json
import logging
import os
import sys
from functools import partial
from typing import Optional

from colorlog import ColoredFormatter
from hyperqueue import Client, Job

from world_model.opendv import create_tokens


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


logger = logging.getLogger("Python HQ Client")


def _path(path: str) -> str:
    path = os.path.expanduser(os.path.expandvars(path))
    return path


parser = argparse.ArgumentParser(description="OpenDV Token Creator")
parser.add_argument("--video_list", type=_path, help="Path to the video list (json)")
parser.add_argument("--metadata", type=_path, help="Path to the metadata file")
parser.add_argument("--outdir", type=_path, help="Output directory")
parser.add_argument("--tmpdir", type=_path, help="Temporary directory")
parser.add_argument("--tokenizer_jit_path", type=_path, help="Path to the tokenizer jit model")
parser.add_argument("--num_writer_threads", type=int, help="Number of threads for writing to disk")
parser.add_argument("--frames_queue_size", type=int, help="Size of the frames queue")
parser.add_argument("--writer_queue_size", type=int, help="Size of the writer queue")
parser.add_argument("--batch_size", type=int, help="Batch size for tokenization")
parser.add_argument("--target_frame_rate", type=int, help="Target frame rate for ffmpeg")
parser.add_argument("--target_width", type=int, help="Target width for resizing")
parser.add_argument("--target_height", type=int, help="Target height for resizing")
parser.add_argument("--keep_temp_frames", action="store_true", help="Keep temporary frames after tokenization")
args = parser.parse_args()

os.makedirs("./logs", exist_ok=True)
os.makedirs("./logs/jobs", exist_ok=True)

setup_logger(logdir="./logs")

# Load the video list
with open(args.video_list, "r") as f:
    video_list = json.load(f)

logger.info(f"Number of videos: {len(video_list)}")

partial_create_tokens = partial(
    create_tokens,
    device="cuda",
    metadata=args.metadata,
    outdir=args.outdir,
    tmpdir=args.tmpdir,
    tokenizer_jit_path=args.tokenizer_jit_path,
    num_writer_threads=args.num_writer_threads,
    frames_queue_size=args.frames_queue_size,
    writer_queue_size=args.writer_queue_size,
    batch_size=args.batch_size,
    target_frame_rate=args.target_frame_rate,
    target_width=args.target_width,
    target_height=args.target_height,
    remove_temp_frames=not args.keep_temp_frames,
)

# for video in video_list:
#     video_id = os.path.basename(video).split(".")[0]
#     logger.info(f"Processing video: {video_id}")

#     sucess = partial_create_tokens(video=video)
#     if not sucess:
#         logger.error(f"Failed to process video: {video_id}")
#         sys.exit(1)

server_dir = os.path.join(os.path.expanduser("~"), ".hq-server")
client = Client(server_dir)

job = Job()

for idx, video in enumerate(video_list):
    video_id = os.path.basename(video).split(".")[0]

    job.function(
        partial_create_tokens,
        kwargs=(dict(video=video)),
        stdout=os.path.join("./logs", "jobs", f"{idx:06d}_{video_id}.log"),
        stderr=os.path.join("./logs", "jobs", f"{idx:06d}_{video_id}.log"),
    )

submitted = client.submit(job)
client.wait_for_jobs([submitted])
