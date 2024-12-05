import logging
import os
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Generator, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch import Tensor
from tqdm import tqdm

Kwargs = Dict[str, Any]
Data = Any
DataPoints = List[Any]


@dataclass
class VideoInfo:
    path: str
    video_id: str
    discard_start: int
    discard_end: int
    duration: int
    end: int
    expected_frames: int


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


def get_duration_from_path(video_path: str) -> float:
    # $(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 !{input_video})
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    duration = subprocess.check_output(cmd).decode("utf-8").strip()
    return float(duration)


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


def preprocess_batch_of_frames(frames: List[str], device: str) -> Tensor:
    frames = [Image.open(frame).convert("RGB") for frame in frames]

    def transform(frame: Image.Image) -> Tensor:
        frame = TF.to_image(frame)
        frame = TF.to_dtype(frame, torch.uint8, scale=True)
        frame = TF.to_dtype(frame, torch.float32, scale=True)
        frame = 2.0 * frame - 1.0
        return frame

    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    return frames.to(device, non_blocking=True)


class TokenCreator:

    def __init__(
        self,
        *,
        video: str,
        device: str,
        metadata: str,
        tmpdir: str,
        outdir: str,
        tokenizer_jit_path: str,
        frames_queue_size: int,
        writer_queue_size: int,
        batch_size: int,
        target_frame_rate: int,
        target_width: int,
        target_height: int,
        num_writer_threads: int = 1,
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
        self.video_path = video
        self.outdir = Path(outdir)
        self.tmpdir = Path(tmpdir)
        self.device = device

        self.logger = logging.getLogger(os.path.basename(video))

        self.remove_temp_frames = remove_temp_frames

        assert num_writer_threads > 0, "num_writer_threads must be positive"

        self.num_writer_threads = num_writer_threads

        # Create output directory
        self.outdir.mkdir(exist_ok=True, parents=True)
        # Create temporary directory
        self.tmpdir.mkdir(exist_ok=True, parents=True)

        # Initialize queues
        self.frames_queue = Queue(maxsize=frames_queue_size)
        self.writer_queue = Queue(maxsize=writer_queue_size)

        # Target frame rate for frames extraction
        self.processing_config = ProcessingConfig(
            fps=target_frame_rate,
            width=target_width,
            height=target_height,
        )

        # Load the videos
        _ = self.format_metadata(metadata)
        meta = self.metadata[os.path.basename(video)]
        video_id = os.path.basename(video).split(".")[0]
        self.video = VideoInfo(path=video, video_id=video_id, **meta)
        self.video.discard_start = 10
        self.video.end = 63
        self.video.expected_frames = 50 * 5

        # Batch size for the tokenizer
        self.batch_size = batch_size

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
        self.tokenizer.to(self.device, non_blocking=True)

        # Progress bars for processing and writing
        self.pbar_frames = None
        self.pbar_tokens = None
        self.pbar_write = None

    def format_metadata(self, metadata: str) -> None:
        db = pd.read_csv(metadata, sep="\t")
        self.metadata = defaultdict(dict)
        for _, row in db.iterrows():
            video_id = row["video_id"].replace("@", "-") + "." + row["container"]
            self.metadata[video_id]["discard_start"] = row["discard_start"]
            self.metadata[video_id]["discard_end"] = row["discard_end"]
            duration = row["duration"]
            if np.isnan(duration):
                duration = get_duration_from_path(self.video_path)

            self.metadata[video_id]["duration"] = duration - row["discard_start"] - row["discard_end"]
            self.metadata[video_id]["end"] = duration - row["discard_end"]
            self.metadata[video_id]["expected_frames"] = int(self.metadata[video_id]["duration"] * self.processing_config.fps)

    def create_frames_from_video(self) -> None:
        """Extract frames from video using PyAV."""
        config = self.processing_config

        try:
            frames_outdir = self.tmpdir / Path(self.video.path).stem
            frames_outdir.mkdir(exist_ok=True)

            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-ss",
                f"{self.video.discard_start}",
                "-i",
                f"{self.video.path}",
                "-to",
                f"{self.video.end}",
                "-vf",
                f"fps={config.fps},scale={config.width}:{config.height}",
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

            with ffmpeg_process(cmd) as process:
                processed_frames = set()
                no_new_frames_count = 0

                while True:
                    new_frames = set(list(frames_outdir.glob("*.jpg"))) - processed_frames

                    if not new_frames:
                        # If no new frames were found, check if the process is still running
                        if process.poll() is not None:
                            # Process has finished, exit if we've checked a few times with no new frames
                            no_new_frames_count += 1
                            if no_new_frames_count >= 3:  # Check 3 times to be sure
                                break
                        time.sleep(1.0)
                        continue

                    # Reset the counter since we found new frames
                    no_new_frames_count = 0
                    processed_frames.update(new_frames)

                    # Update progress
                    with self.lock:
                        self.frames_created += len(new_frames)
                        self.pbar_frames.update(len(new_frames))

                    for frame in new_frames:
                        self.frames_queue.put(frame)

        except Exception as e:
            self.logger.error(f"Error extracting frames from {self.video.path}: {str(e)}")

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
                    batch = preprocess_batch_of_frames(current_batch, self.device)
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
                self.logger.error(f"Error tokenizing frames: {str(e)}")

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
                self.logger.error(f"Error saving token {path}: {str(e)}")

            self.writer_queue.task_done()
            # Update progress
            with self.lock:
                self.written_tokens += 1
                self.pbar_write.total = self.frames_created
                self.pbar_write.update(1)

    def create_opendv_dataset(self) -> bool:
        """Create the dataset using concurrent processing and writing with ThreadPoolExecutor."""
        # Initialize progress bars
        self.pbar_frames = tqdm(
            total=self.video.expected_frames, desc=f"[Processing {self.video.video_id}] Extracting frames from videos"
        )
        self.pbar_tokens = tqdm(total=self.video.expected_frames, desc=f"[Processing {self.video.video_id}] Encoding tokens")
        self.pbar_write = tqdm(
            total=self.video.expected_frames, desc=f"[Processing {self.video.video_id}] Writing tokens to disk"
        )

        success = True
        try:
            # Create thread pools for each task type
            with (
                ThreadPoolExecutor(max_workers=1) as frame_executor,
                ThreadPoolExecutor(max_workers=1) as tokenizer_executor,
                ThreadPoolExecutor(max_workers=self.num_writer_threads) as writer_executor,
            ):

                # Submit frame extraction task
                extraction_future = frame_executor.submit(self.create_frames_from_video)

                # Start tokenizer task
                tokenizer_future = tokenizer_executor.submit(self.tokenize_worker)

                # Start writer tasks
                writer_futures = []
                for _ in range(self.num_writer_threads):
                    future = writer_executor.submit(self.save_tokens)
                    writer_futures.append(future)

                # Wait for frame extraction to complete and handle any exceptions
                try:
                    extraction_future.result()  # This will raise any exceptions that occurred
                except Exception as e:
                    self.logger.error(f"Frame extraction failed: {str(e)}")
                # Signal that frame extraction is complete
                self.all_frames_extracted.set()
                self.logger.info("All frames extracted")

                # Wait for tokenizer to complete
                try:
                    tokenizer_future.result()
                except Exception as e:
                    self.logger.error(f"Tokenizer failed: {str(e)}")
                # Signal that tokenization is complete
                self.processing_complete.set()

                # Wait for writers to complete
                for future in as_completed(writer_futures):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Writer failed: {str(e)}")
                        success = False

        except Exception as e:
            self.logger.error(f"Dataset creation failed: {str(e)}")
            success = False

        finally:
            # Cleanup
            self.pbar_frames.close()
            self.pbar_tokens.close()
            self.pbar_write.close()

            # Make sure all event flags are set to prevent hanging
            self.all_frames_extracted.set()
            self.processing_complete.set()
            self.writting_complete.set()

        return success


def create_tokens(**kwargs) -> bool:
    token_creator = TokenCreator(**kwargs)
    success = token_creator.create_opendv_dataset()
    if success:
        token_creator.logger.info(f"Dataset creation for {token_creator.video.video_id} completed successfully")
    else:
        token_creator.logger.error(f"Dataset creation for {token_creator.video.video_id} failed")
    return success
