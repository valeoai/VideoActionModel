import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
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


def preprocess_batch_of_frames(frames: List[str]) -> Tensor:
    return torch.zeros((len(frames), 3, 224, 224))


def extract_frames(
    video: VideoInfo,
    config: ProcessingConfig,
) -> List[str]:
    tmpdir = Path(config.tmpdir) / Path(video.path).stem
    tmpdir.mkdir(exist_ok=True, parents=True)
    cmd = [
        "ffmpeg",
        "-ss",
        f"{video.trim_start}",
        "-i",
        f"{video.path}",
        "-vf",
        f"fps={config.fps},scale={config.width}:{config.height}",
        "-q:v",
        "2",
        f"{str(tmpdir)}/f_%06d.jpg",
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp_frames = sorted(glob(str(tmpdir / "f_*.jpg") + "/*.jpg"))
    except Exception as e:
        tmp_frames = []
        logger.error(f"Error extracting frames from {video.path}: {str(e)}")

    # Remove frames that are before or after the trim points
    trim_start_frames = int(video.trim_start * config.fps)
    trim_end_frames = int(video.trim_end * config.fps)
    frames = []
    for frame in tmp_frames:
        frame_num = int(Path(frame).stem.split("_")[1])
        if trim_start_frames <= frame_num <= trim_end_frames:
            frames.append(frame)

    return frames


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
        self.videos = video_list
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
        self.video_queue = Queue(maxsize=len(self.videos))
        self.frames_queue = Queue(maxsize=frames_queue_size)
        self.writer_queue = Queue(maxsize=writer_queue_size)

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

        # Progress tracking
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

    def create_frames_from_video(self, video: VideoInfo) -> List[str]:
        """
        Extracts a video clip from a file using frame indices and encodes it into a webdataset-compatible format.
        """
        frames = extract_frames(video, self.processing_config)

        # Update progress
        with self.lock:
            self.frames_created += 1
            self.pbar_frames.update(1)

        return frames

    def ffmpeg_worker(self) -> None:
        """Submit clips to the processor thread pool."""
        while True:
            try:
                video = self.video_queue.get()
            except Empty:
                time.sleep(0.1)
                continue

            if video is None:
                break

            try:
                frames = self.create_frames_from_video(video)
                for frame in frames:
                    self.frames_queue.put(frame)
            except Exception as e:
                raise Exception(str(e))

            self.video_queue.task_done()
            # Update progress
            with self.lock:
                self.frames_created += 1
                self.pbar_frames.update(1)

    def tokenize_frames(self, frames: Tensor) -> Tokens:
        """
        Tokenize a list of frames using a pre-trained tokenizer.
        """
        # Tokenize the frames
        tokens = self.tokenizer(frames)

        # Update progress
        with self.lock:
            self.frames_tokenized += 1
            self.pbar_tokens.update(1)

        return [Tokens(data=token, path=Path(f"{frame}.npy")) for frame, token in zip(frames, tokens)]

    def tokenize_worker(self) -> None:
        """Submit tokens to the writer thread pool."""
        current_batch = []
        break_after = False

        while True:
            frames = self.frames_queue.get()
            if frames is None:
                break_after = True

            try:
                if len(current_batch) == self.batch_size or break_after:
                    batch = preprocess_batch_of_frames(frames)
                    tokens = self.tokenize_frames(batch)

                    # Remove frames from the temporary directory
                    for frame in frames:
                        Path(frame).unlink()

                    for token in tokens:
                        self.writer_queue.put(token)

                    current_batch = []

            except Exception as e:
                raise Exception(str(e))

            self.frames_queue.task_done()
            # Update progress
            with self.lock:
                self.frames_tokenized += 1
                self.pbar_tokens.update(1)

            if break_after:
                break

    def save_tokens(
        self,
    ):
        while True:
            item = self.writer_queue.get()
            if item is None:
                break

            token = item
            try:
                token.path.parent.mkdir(parents=True, exist_ok=True)
                np.save(token.path, token.data, allow_pickle=False)
            except Exception as e:
                raise Exception(str(e))

            self.writer_queue.task_done()
            # Update progress
            with self.lock:
                self.written_tokens += 1
                self.pbar_write.update(1)

    def create_webdataset(self) -> Tuple[Dict[str, int], List[str]]:
        """Create the dataset using concurrent processing and writing."""
        # Initialize progress bars
        self.pbar_frames = tqdm(total=len(self.videos), desc="[OpenDV preprocess] Extracting frames from videos")
        self.pbar_tokens = tqdm(desc="[OpenDV preprocess] Encoding tokens")
        self.pbar_write = tqdm(desc="[OpenDV preprocess] Writing tokens to disk")

        # Create save threads
        ffmpeg_threads = []
        for i in range(self.num_ffmpeg_threads):
            thread = Thread(target=self.ffmpeg_worker)
            thread.start()
            ffmpeg_threads.append(thread)

        # Create the tokenizer thread
        tokenizer_thread = Thread(target=self.tokenize_worker)
        tokenizer_thread.start()

        # Create the writer thread
        writer_threads = []
        for i in range(self.num_writer_threads):
            thread = Thread(target=self.save_tokens)
            thread.start()
            writer_threads.append(thread)

        # Sending None to Threads is an exit signal
        for _ in range(self.num_ffmpeg_threads):
            self.video_queue.put(None)
        # blocks the main thread until the all thread has completed
        for thread in ffmpeg_threads:
            thread.join()

        self.frames_queue.put(None)
        tokenizer_thread.join()

        for _ in range(self.num_writer_threads):
            self.writer_queue.put(None)
        for thread in writer_threads:
            thread.join()

        self.pbar_frames.close()
        self.pbar_tokens.close()
        self.pbar_write.close()

        # Write dataset info
        dataset_info = {
            "frames_created": self.frames_created,
            "frames_tokenized": self.frames_tokenized,
            "written_tokens": self.written_tokens,
        }

        with open(self.outdir / f"rank-{self.rank}-dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=4)

        return dataset_info


# # Example usage:
# if __name__ == "__main__":
#     import wm_lib.logging

#     wm_lib.logging.setup_logger()

#     # Example clips list
#     with open("/home/eramzi/iveco/datasets_iveco_raw/OpenDV_Youtube/shuffled_clips.json", "rb") as f:
#         all_clips = json.load(f)

#     clips = all_clips[:100]

#     # Create dataset with separate thread pools
#     creator = WebdatasetWriter(
#         task="images_from_videos",
#         data_points=clips,
#         outdir="/home/eramzi/data/shards_2",
#         rank=0,
#         tmpdir="/home/eramzi/data/tmp_2",
#         num_processor_threads="auto",  # Adjust based on CPU cores
#         num_writer_threads="auto",  # Usually fewer writers than processors
#         max_samples_per_shard=10,
#         queue_size=100000,  # Adjust based on memory availability
#         batch_size=None,
#         resize=(576, 288),
#         target_frame_rate=10,
#     )

#     # Create the dataset
#     info, _ = creator.create_webdataset()
#     print("Dataset creation completed!")
#     print(f"Processed {info['processed_clips']} clips")
#     print(f"Written {info['written_clips']} clips")
#     print(f"Created {info['num_shards']} shards")
