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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

Kwargs = Dict[str, Any]
Sample = Dict[str, Any]
Batch = Dict[str, Any]

logger = logging.getLogger("world_model/TokenCreator")


PARAMS = {
    "opendv": {"resize_factor": 1.0},
    "nuplan": {"resize_factor": 3.75},
    "nuscenes": {"resize_factor": 3.125},
}


@dataclass
class Tokens:
    data: np.ndarray
    path: Path


def resize_by_factor(img: Tensor, resize_factor: float) -> Tensor:
    new_width = int(img.shape[2] / resize_factor)
    new_height = int(img.shape[1] / resize_factor)
    return TF.resize(img, (new_height, new_width), antialias=True)


class FramesDataset(Dataset):
    """Vanilla Dataset for loading frames from a list of file paths."""

    def __init__(self, dataset: str, frames_file_list: List[str]) -> None:
        self.dataset = dataset
        self.frames_file_list = frames_file_list

    def __len__(self) -> int:
        return len(self.frames_file_list)

    def __getitem__(self, idx: int) -> Sample:
        path = self.frames_file_list[idx]
        frame = Image.open(path).convert("RGB")
        frame = TF.to_image(frame)
        frame = TF.to_dtype(frame, torch.uint8, scale=True)
        frame = resize_by_factor(frame, PARAMS[self.dataset]["resize_factor"])
        frame = TF.to_dtype(frame, torch.float32, scale=True)
        frame = 2.0 * frame - 1.0
        return {"image": frame, "path": path}


class TokenCreator:

    def __init__(
        self,
        *,
        dataset: str,
        frames: str,
        outdir: str,
        tokenizer_jit_path: str,
        batch_size: int,
        writer_queue_size: int = 1000,
        num_writer_threads: int = 5,
        num_workers: int = 8,
        dtype: str = "bf16",
        device: str = "cuda",
    ) -> None:
        """
        Initialize the dataset creator with separate processing and writing threads.

        Parameters
        ----------
        dataset: str; dataset to process
        frames: str; path to a file containing a list of frames to tokenize
        device: str; device to use for processing
        outdir: str; path to the output directory
        tokenizer_jit_path: str; path to the pre-trained tokenizer model
        writer_queue_size: int; size of the writer queue
        batch_size: int; batch size for processing
        num_writer_threads: int; number of writer threads to use
        """
        assert dataset in [
            "opendv",
            "nuplan",
            "nuscenes",
        ], f"Invalid dataset: {dataset}, must be one of 'opendv', 'nuplan', 'nuscenes'"
        self.dataset = dataset
        self.outdir = Path(outdir)
        self.device = device
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]

        self.num_writer_threads = num_writer_threads
        assert num_writer_threads > 0, "num_writer_threads must be positive"

        # Create output directory
        self.outdir.mkdir(exist_ok=True, parents=True)

        # Initialize queues
        with open(frames, "r") as f:
            self.frames = [x.strip().replace("\n", "") for x in f.readlines()]
        self.job_name = f"{len(self.frames)} frames"
        self.number_of_frames = len(self.frames)
        dataset = FramesDataset(self.dataset, self.frames)
        self.loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

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

    def token_path(self, path: str) -> Path:
        if self.dataset == "opendv":
            # frames have the following path
            # DISK_DIR / {CHANNEL} / {VIDEO_ID} / f_{frame_num}.jpg
            # Tokens should have the following path
            # self.outdir / {CHANNEL} / {VIDEO_ID} / t_{frame_num}.npy
            return (
                self.outdir
                / Path(path).parent.parent.stem
                / Path(path).parent.stem
                / f"{Path(path).stem}.npy".replace("f_", "t_")
            )
        elif self.dataset == "nuplan":
            # frames have the following path
            # DISK_DIR / {CAMERA_NUM} / {DATES} / {CAMERA} / {id}.jpg
            # Tokens should have the following path
            # self.outdir / {DATES} / {CAMERA} / {id}.npy
            return self.outdir / Path("/".join(str(Path(path).parent).split("/")[-2:])) / f"{Path(path).stem}.npy"
        elif self.dataset == "nuscenes":
            # frames have the following path
            # DISK_DIR / {CAMERA} / {TIMESTAMP}.jpg
            # Tokens should have the following path
            # self.outdir / {CAMERA} / {TIMESTAMP}.npy
            return self.outdir / Path(path).parent.stem / f"{Path(path).stem}.npy"

    def tokenize_frames(self, frames: Batch) -> List[Tokens]:
        """
        Tokenize a list of frames using a pre-trained tokenizer.
        """
        # Tokenize the frames
        with torch.amp.autocast(self.device, dtype=self.dtype):
            tokens = self.tokenizer(frames["image"].to(self.device, non_blocking=True))

        out_tokens = []
        for path, token in zip(frames["path"], tokens):
            token_path = self.token_path(path)
            out_tokens.append(Tokens(data=token.cpu().numpy(), path=token_path))

        return out_tokens

    def tokenize_worker(self) -> bool:
        """Submit tokens to the writer thread pool."""
        success = True
        try:
            for batch in self.loader:
                tokens = self.tokenize_frames(batch)

                # Update progress
                with self.lock:
                    self.frames_tokenized += len(batch["path"])
                    self.pbar_tokens.update(len(batch["path"]))

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
