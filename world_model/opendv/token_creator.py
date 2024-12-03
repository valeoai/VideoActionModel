import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from colorlog import ColoredFormatter
from PIL import Image
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger("OpenDV")
Kwargs = Dict[str, Any]
Data = Any
DataPoints = List[Any]


logger.handlers.clear()
formatter = ColoredFormatter(
    "[%(cyan)s%(asctime)s%(reset)s]" "[%(light_blue)s%(name)s%(reset)s]" "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
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
logger.addHandler(handler)


@dataclass
class VideoInfo:
    path: str
    trim_start: int
    trim_end: int
    duration: int = None


@dataclass
class ProcessingConfig:
    fps: int
    width: int
    height: int


@dataclass
class Tokens:
    data: np.ndarray
    path: Path


@dataclass
class Frame:
    data: Image.Image
    path: str


@dataclass
class FramesBatch:
    data: Tensor
    paths: List[str]


@contextmanager
def ffmpeg_process(cmd: List[str]) -> Generator[subprocess.Popen, None, None]:
    process = subprocess.Popen(cmd)
    try:
        yield process
    finally:
        process.terminate()
        process.wait(timeout=5)  # Wait up to 5 seconds for graceful termination
        try:
            process.kill()  # Force kill if still running
        except ProcessLookupError:
            pass  # Process already terminated


def preprocess_batch_of_frames(frames: List[str]) -> Tensor:
    frames = [Image.open(frame).convert("RGB") for frame in frames]

    def transform(frame: Image.Image) -> Tensor:
        frame = TF.to_image(frame)
        frame = TF.to_dtype(frame, torch.uint8, scale=True)
        frame = TF.to_dtype(frame, torch.float32, scale=True)
        frame = 2.0 * frame - 1.0
        return frame

    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    return frames.to("cuda", non_blocking=True)


class TokenCreator:

    def __init__(
        self,
        *,
        video_list: List[str],
        metadata: pd.DataFrame,
        tmpdir: str,
        outdir: str,
        rank: int,
        tokenizer_jit_path: str,
        num_frames_threads: int,
        num_writer_threads: int,
        frames_queue_size: int,
        writer_queue_size: int,
        batch_size: int,
        target_frame_rate: int,
        target_width: int,
        target_height: int,
        remove_temp_frames: bool = True,
    ) -> None:
        """
        Initialize the dataset creator with separate processing and writing threads.

        Args:
            task: Task name for the webdataset formatter
            data_points: List of data points to be sharded
            outdir: Output directory
            rank: Rank of the process for creating unique file names
            tmpdir: Temporary folder for storing video clips
            num_processor_threads: Number of threads for video processing
            num_writer_threads: Number of threads for writing to tar files
            max_samples_per_shard: Number of samples per shard
            queue_size: Size of the queue connecting processors and writers
            check_kill_signal: Function to check if the process should terminate
            preprocessor_kwargs: Additional arguments for the preprocessor
        """
        self.video_list = video_list
        self.outdir = Path(outdir)
        self.tmpdir = Path(tmpdir)
        self.rank = rank

        self.remove_temp_frames = remove_temp_frames

        assert num_frames_threads > 0, "num_processor_threads must be positive"
        assert num_writer_threads > 0, "num_writer_threads must be positive"

        self.num_frames_threads = num_frames_threads
        self.num_writer_threads = num_writer_threads

        # Create output directory
        self.outdir.mkdir(exist_ok=True, parents=True)
        # Create temporary directory
        self.tmpdir.mkdir(exist_ok=True, parents=True)

        # Initialize queues
        self.frames_queue = Queue(maxsize=frames_queue_size)
        self.writer_queue = Queue(maxsize=writer_queue_size)

        # Load the videos
        _ = self.format_metadata(metadata)
        self.videos = []
        for video_pth in self.video_list:
            meta = self.metadata[os.path.basename(video_pth)]
            video = VideoInfo(
                path=video_pth,
                trim_start=meta["trim_start"],
                trim_end=meta["trim_end"],
                duration=meta["duration"],
            )
            self.videos.append(video)

        # Batch size for the tokenizer
        self.batch_size = batch_size

        # Target frame rate for frames extraction
        self.processing_config = ProcessingConfig(
            fps=target_frame_rate,
            width=target_width,
            height=target_height,
        )

        # Flag to signal when processing is complete
        self.processing_complete = threading.Event()
        self.all_frames_extracted = threading.Event()
        self.writting_complete = threading.Event()

        # Progress tracking
        self.video_processed = 0
        self.frames_created = 0
        self.frames_tokenized = 0
        self.written_tokens = 0
        self.lock = threading.Lock()

        # Load the tokenizer
        self.tokenizer = torch.jit.load(tokenizer_jit_path)
        self.tokenizer.eval()
        self.tokenizer.cuda()

        # Progress bars for processing and writing
        self.pbar_frames = None
        self.pbar_tokens = None
        self.pbar_write = None

    def format_metadata(self, db: pd.DataFrame) -> None:
        self.metadata = defaultdict(dict)
        for _, row in db.iterrows():
            video_id = row["video_id"].replace("@", "-") + "." + row["container"]
            self.metadata[video_id]["trim_start"] = row["discard_start"]
            self.metadata[video_id]["trim_end"] = row["discard_end"]
            if not np.isnan(duration := row["duration"]):
                self.metadata[video_id]["duration"] = duration - row["discard_start"] - row["discard_end"]
            else:
                self.metadata[video_id]["duration"] = None

    def create_frames_from_video(
        self,
        video: VideoInfo,
    ) -> None:
        """Extract frames from video using PyAV."""
        config = self.processing_config

        try:
            frames_outdir = self.tmpdir / Path(video.path).stem
            frames_outdir.mkdir(exist_ok=True)

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-ss",
                "10",
                "-i",
                f"{video.path}",
                "-t",
                f"{video.duration}",
                "-vf",
                f"fps={config.fps},scale={config.width}:{config.height}",
                "-threads",
                "2",  # Use all available threads
                "-q:v",
                "2",
                "-vsync",
                "0",  # Disable video sync to speed up processing
                "-frame_pts",
                "0",  # Don't write presentation timestamp
                "-movflags",
                "+faststart",
                "-loglevel",
                "error",  # Add this line to suppress ffmpeg logs
                frames_outdir / "f_%06d.jpg",
            ]

            with ffmpeg_process(cmd):
                processed_frames = set()

                while True:
                    new_frames = set(list(frames_outdir.glob("*.jpg"))) - processed_frames
                    processed_frames.update(new_frames)

                    # Update progress
                    with self.lock:
                        self.frames_created += len(new_frames)
                        self.pbar_frames.set_postfix({"frames_created": self.frames_created})

                    for frame in new_frames:
                        self.frames_queue.put(frame)
                    time.sleep(1.0)

        except Exception as e:
            logger.error(f"Error extracting frames from {video.path}: {str(e)}")

    def tokenize_frames(self, frames: FramesBatch) -> Tokens:
        """
        Tokenize a list of frames using a pre-trained tokenizer.
        """
        # Tokenize the frames
        try:
            tokens = self.tokenizer(frames.data)
        except Exception as e:
            raise Exception(str(e))

        # frames have the following path
        # self.tmpdir / Path(video.path).stem / f_{frame_num}.jpg
        # Tokens should have the following path
        # self.outdir / Path(video.path).stem / f_{frame_num}.npy
        out_tokens = []
        for path, token in zip(frames.paths, tokens):
            token_path = self.outdir / Path(path).parent.stem / f"{Path(path).stem}.npy"
            out_tokens.append(Tokens(data=token.cpu().numpy(), path=token_path))

        return out_tokens

    def tokenize_worker(self) -> None:
        """Submit tokens to the writer thread pool."""
        current_batch = []
        timeout = 1.0

        while not self.processing_complete.is_set():
            try:
                # Try to fill batch with timeout
                while len(current_batch) < self.batch_size:
                    try:
                        frames = self.frames_queue.get(timeout=timeout)
                        current_batch.append(frames)
                        self.frames_queue.task_done()
                    except Empty:
                        # If queue is empty and extraction is done, process remaining frames
                        if self.all_frames_extracted.is_set() and current_batch:
                            break
                        elif self.all_frames_extracted.is_set():
                            self.processing_complete.set()
                            return
                        time.sleep(1.0)
                        continue

                if (len(current_batch) == self.batch_size) or (self.all_frames_extracted.is_set() and len(current_batch) > 0):
                    batch = preprocess_batch_of_frames(current_batch)
                    tokens = self.tokenize_frames(FramesBatch(data=batch, paths=current_batch))

                    # Update progress
                    with self.lock:
                        self.frames_tokenized += len(current_batch)
                        self.pbar_tokens.total = self.frames_created
                        self.pbar_tokens.update(len(current_batch))

                    for token in tokens:
                        self.writer_queue.put(token)

                    current_batch = []

            except Exception as e:
                current_batch = []
                logger.error(f"Error tokenizing frames: {str(e)}")

    def save_tokens(self) -> None:
        while not self.writting_complete.is_set():
            try:
                token = self.writer_queue.get(timeout=5.0)
            except Empty:
                if self.processing_complete.is_set():
                    self.writting_complete.set()
                    return
                time.sleep(1.0)
                continue

            try:
                path = Path(token.path)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, token.data.astype(np.uint16))
            except Exception as e:
                logger.error(f"Error saving token {path}: {str(e)}")

            self.writer_queue.task_done()
            # Update progress
            with self.lock:
                self.written_tokens += 1
                self.pbar_write.total = self.frames_created
                self.pbar_write.update(1)

    def create_opendv_dataset(self) -> Tuple[Dict[str, int], List[str]]:
        """Create the dataset using concurrent processing and writing with ThreadPoolExecutor."""
        # Initialize progress bars
        self.pbar_frames = tqdm(total=len(self.videos), desc="[OpenDV preprocess] Extracting frames from videos")
        self.pbar_tokens = tqdm(total=9e9, desc="[OpenDV preprocess] Encoding tokens")
        self.pbar_write = tqdm(total=9e9, desc="[OpenDV preprocess] Writing tokens to disk")

        try:
            # Create thread pools for each task type
            with (
                ThreadPoolExecutor(max_workers=self.num_frames_threads) as frame_executor,
                ThreadPoolExecutor(max_workers=1) as tokenizer_executor,
                ThreadPoolExecutor(max_workers=self.num_writer_threads) as writer_executor,
            ):

                # Submit frame extraction tasks
                frame_futures = []
                for video in self.videos:
                    future = frame_executor.submit(self.create_frames_from_video, video)
                    frame_futures.append(future)

                # Start tokenizer task
                tokenizer_future = tokenizer_executor.submit(self.tokenize_worker)

                # Start writer tasks
                writer_futures = []
                for _ in range(self.num_writer_threads):
                    future = writer_executor.submit(self.save_tokens)
                    writer_futures.append(future)

                # Wait for frame extraction to complete and handle any exceptions
                for future in as_completed(frame_futures):
                    try:
                        future.result()  # This will raise any exceptions that occurred
                    except Exception as e:
                        logger.error(f"Frame extraction failed: {str(e)}")

                # Signal that frame extraction is complete
                self.all_frames_extracted.set()
                logger.info("All frames extracted")

                # Wait for tokenizer to complete
                try:
                    tokenizer_future.result()
                except Exception as e:
                    logger.error(f"Tokenizer failed: {str(e)}")

                # Signal that tokenization is complete
                self.processing_complete.set()

                # Wait for writers to complete
                for future in as_completed(writer_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Writer failed: {str(e)}")

        except Exception as e:
            logger.error(f"Dataset creation failed: {str(e)}")
            raise

        finally:
            # Cleanup
            self.pbar_frames.close()
            self.pbar_tokens.close()
            self.pbar_write.close()

            # Make sure all event flags are set to prevent hanging
            self.all_frames_extracted.set()
            self.processing_complete.set()
            self.writting_complete.set()

        # Write dataset info
        dataset_info = {
            "frames_created": self.frames_created,
            "frames_tokenized": self.frames_tokenized,
            "written_tokens": self.written_tokens,
        }

        with open(self.outdir / f"rank-{self.rank}-dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=4)

        return dataset_info
