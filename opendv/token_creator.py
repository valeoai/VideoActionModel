import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Tuple

import av
import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger("OpenDV")
Kwargs = Dict[str, Any]
Data = Any
DataPoints = List[Any]


@dataclass
class VideoInfo:
    path: str
    trim_start: int
    trim_end: int


@dataclass
class ProcessingConfig:
    tmpdir: str
    fps: int
    width: int
    height: int


@dataclass
class Tokens:
    data: np.ndarray
    path: Path


@dataclass
class FramesBatch:
    data: Tensor
    paths: List[str]


def preprocess_batch_of_frames(frames: List[str]) -> Tensor:
    frames = [np.array(Image.open(frame).convert("RGB")) for frame in frames]

    def transform(frame):
        frame = TF.to_image(frame)
        frame = TF.to_dtype(frame, torch.uint8, scale=True)
        frame = TF.to_dtype(frame, torch.float32, scale=True)
        frame = TF.normalize(frame, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        return frame

    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    return frames.to("cuda", non_blocking=True)


class TokenCreator:

    def __init__(
        self,
        *,
        video_list: str,
        outdir: str,
        rank: int,
        tmpdir: str,
        tokenizer_jit_path: str,
        num_ffmpeg_threads: int,
        num_writer_threads: int,
        frames_queue_size: int,
        writer_queue_size: int,
        batch_size: int,
        target_frame_rate: int,
        target_width: int,
        target_height: int,
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
        self.rank = rank
        self.tmpdir = Path(tmpdir)

        assert num_ffmpeg_threads > 0, "num_processor_threads must be positive"
        assert num_writer_threads > 0, "num_writer_threads must be positive"

        self.num_ffmpeg_threads = num_ffmpeg_threads
        self.num_writer_threads = num_writer_threads

        # Create output directory
        self.outdir.mkdir(exist_ok=True, parents=True)
        # Temporary folder for storing video clips
        self.tmpdir.mkdir(exist_ok=True, parents=True)

        # Initialize queues
        self.video_queue = Queue(maxsize=len(self.video_list))
        self.frames_queue = Queue(maxsize=frames_queue_size)
        self.writer_queue = Queue(maxsize=writer_queue_size)

        # Load the videos
        self.videos = [VideoInfo(path=video, trim_start=0, trim_end=10) for video in self.video_list]
        for video in self.videos:
            self.video_queue.put(video)

        # Batch size for the tokenizer
        self.batch_size = batch_size

        # Target frame rate for FFMPEG
        self.processing_config = ProcessingConfig(
            tmpdir=tmpdir,
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
        self.ffmpeg_processes = []

        # Load the tokenizer
        self.tokenizer = torch.jit.load(tokenizer_jit_path)
        self.tokenizer.eval()
        self.tokenizer.cuda()

        # Progress bars for processing and writing
        self.pbar_frames = None
        self.pbar_tokens = None
        self.pbar_write = None

    def create_frames_from_video(
        self,
        video: VideoInfo,
    ) -> None:
        """Extract frames from video using PyAV."""
        config = self.processing_config
        tmpdir = Path(config.tmpdir) / Path(video.path).stem
        tmpdir.mkdir(exist_ok=True, parents=True)

        try:
            container = av.open(video.path)
            stream = container.streams.video[0]

            # Set the timebase and start time
            # stream.codec_context.skip_frame = "NONKEY"  # Only decode keyframes for speed
            # start_pts = int(video.trim_start * stream.time_base.denominator)
            # container.seek(start_pts)

            # Calculate frame intervals for desired FPS
            source_fps = float(stream.average_rate)
            frame_interval = source_fps / config.fps
            next_frame_idx = 0.0
            frame_count = 0

            for frame_idx, frame in enumerate(container.decode(video=0)):
                if frame_idx < next_frame_idx:
                    continue

                # # Check if we've reached the end time
                # frame_time = frame.pts * float(stream.time_base)
                # if frame_time > video.trim_end:
                #     break

                # Save the frame
                frame_path = tmpdir / f"f_{frame_count:06d}.jpg"
                img = frame.to_image()

                # Resize if needed
                if (img.width != config.width) or (img.height != config.height):
                    img = img.resize((config.width, config.height), Image.Resampling.LANCZOS)

                img.save(str(frame_path), quality=95)
                self.frames_queue.put(str(frame_path))

                # Update progress
                with self.lock:
                    self.frames_created += 1
                    if self.pbar_frames is not None:
                        self.pbar_frames.set_postfix({"frames_created": self.frames_created})
                        self.pbar_frames.refresh()

                frame_count += 1
                next_frame_idx += frame_interval

                # Optional: add check for early termination
                if frame_count >= 100:  # Keep your original limit if needed
                    break

        except Exception as e:
            logger.error(f"Error extracting frames from {video.path}: {str(e)}")
        finally:
            container.close()

    def ffmpeg_worker(self) -> None:
        """Submit clips to the processor thread pool."""
        while True:
            try:
                video = self.video_queue.get()
            except Empty:
                time.sleep(1)
                self.all_frames_extracted.set()
                break

            try:
                _ = self.create_frames_from_video(video)
            except Exception as e:
                logger.error(f"Error processing video {video.path}: {str(e)}")

            self.video_queue.task_done()
            # Update progress
            with self.lock:
                self.video_processed += 1
                self.pbar_frames.update(1)

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
            # Path(path).unlink()

        return out_tokens

    def tokenize_worker(self) -> None:
        """Submit tokens to the writer thread pool."""
        current_batch = []
        timeout = 1.0

        while not self.processing_complete.is_set():
            print(self.frames_queue.qsize())
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
                        self.pbar_tokens.refresh()
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
                np.save(path, token.data, allow_pickle=False)
            except Exception as e:
                logger.error(f"Error saving token {path}: {str(e)}")

            self.writer_queue.task_done()
            # Update progress
            with self.lock:
                self.written_tokens += 1
                self.pbar_write.refresh()
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
                ThreadPoolExecutor(max_workers=self.num_ffmpeg_threads) as frame_executor,
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


# Example usage:
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    video_list = glob(os.path.expanduser("~/data/*.mp4"))
    video_list = video_list[:1]
    logger.info(f"Found {len(video_list)} videos")
    outdir = os.path.expanduser("~/data/OpenDV_Youtube/tokens")
    rank = 0
    tmpdir = os.path.expanduser("~/data/OpenDV_Youtube/tmp")
    tokenizer_jit_path = os.path.expanduser("~/iveco/scratch_iveco/world_model_JZGC4/jit_models/VQ_ds16_16384_llamagen.jit")

    num_ffmpeg_threads = 1
    num_writer_threads = 1
    frames_queue_size = 100000
    writer_queue_size = 10000
    batch_size = 12
    target_frame_rate = 5
    target_width = 512
    target_height = 288

    token_creator = TokenCreator(
        video_list=video_list,
        outdir=outdir,
        rank=rank,
        tmpdir=tmpdir,
        tokenizer_jit_path=tokenizer_jit_path,
        num_ffmpeg_threads=num_ffmpeg_threads,
        num_writer_threads=num_writer_threads,
        frames_queue_size=frames_queue_size,
        writer_queue_size=writer_queue_size,
        batch_size=batch_size,
        target_frame_rate=target_frame_rate,
        target_width=target_width,
        target_height=target_height,
    )

    token_creator.create_opendv_dataset()
