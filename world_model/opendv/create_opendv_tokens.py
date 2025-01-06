import argparse
import logging
import os
import sys
from functools import partial
from glob import glob
from typing import Optional

from colorlog import ColoredFormatter
from hyperqueue import Client, Job
from hyperqueue.ffi.protocol import ResourceRequest

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
        logfile = os.path.join(logdir, "hq_python.log")
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
parser.add_argument("--server_dir", type=_path, help="Path HQ server directory")
parser.add_argument("--frames_dir", type=_path, required=True, help="Directory for already extracted frames")
parser.add_argument("--outdir", type=_path, help="Output directory")
parser.add_argument("--tokenizer_jit_path", type=_path, help="Path to the tokenizer jit model")
parser.add_argument("--num_writer_threads", type=int, help="Number of threads for writing to disk")
parser.add_argument("--writer_queue_size", type=int, help="Size of the writer queue")
parser.add_argument("--num_cpus", type=int, required=True, help="Number of CPUs")
parser.add_argument("--batch_size", type=int, help="Batch size for tokenization")
parser.add_argument("--dtype", type=str, default="bf16", help="Data type for tokenization", choices=["bf16", "fp16", "fp32"])
args = parser.parse_args()
setup_logger(logdir="./hq_tokenize_opendv")

logger.info("Creating hyperqueue client")
client = Client(server_dir=args.server_dir)

frames_file_list = sorted(glob(os.path.join(args.frames_dir, "*.txt"), recursive=True))
logger.info(f"Number of job to create: {len(frames_file_list)}")

partial_create_tokens = partial(
    create_tokens,
    device="cuda",
    outdir=args.outdir,
    tokenizer_jit_path=args.tokenizer_jit_path,
    num_writer_threads=args.num_writer_threads,
    writer_queue_size=args.writer_queue_size,
    batch_size=args.batch_size,
    num_workers=args.num_cpus,
    dtype=args.dtype,
)

job = Job()
for idx, frame_file in enumerate(frames_file_list):
    # We create an independent job for each chunk of files.
    # We use chunks of files so that if a job fails, we don't have to reprocess all the files.
    logger.info(f"Creating task {idx:06d} for {frame_file}")
    job.function(
        partial_create_tokens,
        kwargs={"frames": frame_file},
        stdout=os.path.join("./hq_tokenize_opendv", "tokens", f"{idx:06d}.out"),
        stderr=os.path.join("./hq_tokenize_opendv", "tokens", f"{idx:06d}.err"),
        resources=ResourceRequest(  # We need to provide the resources to prevent the job from running on the same GPU
            cpus=args.num_cpus,
            resources={"gpus/nvidia": 1},
        ),
    )

submitted = client.submit(job)
logger.info(f"Job ID: {submitted}")
logger.info("Waiting for jobs to finish")
client.wait_for_jobs([submitted])
