import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch import Tensor
from tqdm import tqdm

Kwargs = Dict[str, Any]

logger = logging.getLogger("world_model/TokenCreator")


@dataclass
class Tokens:
    data: np.ndarray
    path: Path


@dataclass
class FramesBatch:
    data: Tensor
    paths: List[str]


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
        frames: str,
        outdir: str,
        tokenizer_jit_path: str,
        batch_size: int,
        writer_queue_size: int = 1000,
        num_writer_threads: int = 5,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the dataset creator with separate processing and writing threads.

        Parameters
        ----------
        frames: str; path to a file containing a list of frames to tokenize
        device: str; device to use for processing
        outdir: str; path to the output directory
        tokenizer_jit_path: str; path to the pre-trained tokenizer model
        writer_queue_size: int; size of the writer queue
        batch_size: int; batch size for processing
        num_writer_threads: int; number of writer threads to use
        """
        self.outdir = Path(outdir)
        self.device = device

        self.num_writer_threads = num_writer_threads
        assert num_writer_threads > 0, "num_writer_threads must be positive"

        # Create output directory
        self.outdir.mkdir(exist_ok=True, parents=True)

        # Initialize queues
        with open(frames, "r") as f:
            self.frames = [x.strip().replace("\n", "") for x in f.readlines()]
        self.job_name = f"{len(self.frames)} frames"
        self.number_of_frames = len(self.frames)

        self.writer_queue = Queue(maxsize=writer_queue_size)

        # Batch size for the tokenizer
        self.batch_size = batch_size
        # Flag to signal when processing is complete
        self.processing_complete = threading.Event()
        self.writting_complete = threading.Event()

        # Progress tracking
        self.frames_tokenized = 0
        self.written_tokens = 0
        self.lock = threading.Lock()

        # Load the tokenizer
        self.tokenizer = torch.jit.load(tokenizer_jit_path)
        self.tokenizer.eval()
        self.tokenizer.to(self.device, non_blocking=True)

        # Progress bars for processing and writing
        self.pbar_tokens = None
        self.pbar_write = None

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
        # DISK_DIR / {CHANNEL} / {VIDEO_ID} / f_{frame_num}.jpg
        # Tokens should have the following path
        # self.outdir / {CHANNEL} / {VIDEO_ID} / t_{frame_num}.npy
        out_tokens = []
        for path, token in zip(frames.paths, tokens):
            token_path = (
                self.outdir
                / Path(path).parent.parent.stem
                / Path(path).parent.stem
                / f"{Path(path).stem}.npy".replace("f_", "t_")
            )
            out_tokens.append(Tokens(data=token.cpu().numpy(), path=token_path))

        return out_tokens

    def tokenize_worker(self) -> bool:
        """Submit tokens to the writer thread pool."""
        success = True
        try:
            for start_batch in range(0, len(self.frames), self.batch_size):
                current_batch = self.frames[start_batch : start_batch + self.batch_size]
                batch = preprocess_batch_of_frames(current_batch, self.device)
                tokens = self.tokenize_frames(FramesBatch(data=batch, paths=current_batch))

                # Update progress
                with self.lock:
                    self.frames_tokenized += len(current_batch)
                    self.pbar_tokens.update(len(current_batch))

                for token in tokens:
                    self.writer_queue.put(token)

        except Exception as e:
            success = False
            logger.error(f"Error tokenizing frames: {str(e)}")

        finally:
            # Signal that tokenization is complete
            self.processing_complete.set()

        return success

    def save_tokens(self) -> None:
        while not self.writting_complete.is_set():
            try:
                token = self.writer_queue.get(timeout=1.0)
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

    def create_opendv_dataset(self) -> bool:
        """Create the dataset using concurrent processing and writing with ThreadPoolExecutor."""
        # Initialize progress bars
        self.pbar_tokens = tqdm(total=self.number_of_frames, desc=f"[Processing {self.job_name}] Encoding tokens")
        self.pbar_write = tqdm(total=self.number_of_frames, desc=f"[Processing {self.job_name}] Writing tokens to disk")

        success = True
        try:
            # Create thread pools for each task type
            with (
                ThreadPoolExecutor(max_workers=1) as tokenizer_executor,
                ThreadPoolExecutor(max_workers=self.num_writer_threads) as writer_executor,
            ):

                # Start tokenizer task
                tokenizer_future = tokenizer_executor.submit(self.tokenize_worker)

                # Start writer tasks
                writer_futures = []
                for _ in range(self.num_writer_threads):
                    future = writer_executor.submit(self.save_tokens)
                    writer_futures.append(future)

                # Wait for tokenizer to complete
                success = tokenizer_future.result()
                # Signal that tokenization is complete
                self.processing_complete.set()

                # Wait for writers to complete
                for future in as_completed(writer_futures):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Writer failed: {str(e)}")
                        success = False

        except Exception as e:
            logger.error(f"Dataset creation failed: {str(e)}")
            success = False

        finally:
            # Cleanup
            self.pbar_tokens.close()
            self.pbar_write.close()

            # Make sure all event flags are set to prevent hanging
            self.processing_complete.set()
            self.writting_complete.set()

        return success


def create_tokens(**kwargs: Kwargs) -> bool:
    token_creator = TokenCreator(**kwargs)
    success = token_creator.create_opendv_dataset()
    if success:
        logger.info(f"Dataset creation for {token_creator.job_name} completed successfully")
    else:
        logger.error(f"Dataset creation for {token_creator.job_name} failed")
        raise ValueError(f"Dataset creation for {token_creator.job_name} failed")
    return success
