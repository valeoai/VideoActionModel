"""
This script provides code to generate future frames tokens from a pre-trained world-model.

Outputs follow this structure:

/<output_dir>/
    <model_log_dir name>/
        inference/
            <checkpoint_name>/
                generated_tokens_metadata.yaml
                <top_k,top_p,argmax>_sampling/
                    generation_<index number of generation [0,N-1]>/
                        visual_tokens/
                            samples/
                                CAM_FRONT/
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708429012404.pkl
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708430512404.pkl
                                    ...

                        visual_logits/
                            samples/
                                CAM_FRONT/
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708429012404.pkl
                                    n008-2018-07-27-12-07-38-0400__CAM_FRONT__1532708430512404.pkl
                                    ...

With every pickle file under the `tokens` folder containing:
- visual_tokens of shape [H, W]

With every pickle file under the `logits` (only for generation_0) folder containing:
- visual_logits of shape [vocab_size, H, W]

generated_tokens_metadata.yaml contains:
    - inference_config: the configuration and metadata used to run the inference (e.g., starting index
    of the generation in the sequence, number of context and prediction frames, the
    original path to the model's checkpoint, the inference code git hash ...)
    - training_config: the initial configuration (e.g., hyperparams) and metadata (training code git hash)
    used to train the model.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import mup
import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from lightning.pytorch.callbacks import BasePredictionWriter, Callback
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor

from world_model.utils import RankedLogger
from world_model.utils.generation import GenerationConfig, ImageGenerator, Sampler

log = RankedLogger(__name__, rank_zero_only=True)

Value = Any
Batch = Dict[str, Any]
StateDict = Dict[str, Any]


def worker_rnd_init(x: int) -> None:
    np.random.seed(42 + x)


def get_nested_value(config: Dict[str, Any], key_path: str) -> Value:
    """Helper to safely get nested dictionary values using dot notation."""
    try:
        value = config
        for key in key_path.split("."):
            value = value[key] if isinstance(value, dict) else getattr(value, key)
        return value
    except (KeyError, AttributeError):
        return "NOT FOUND"


def print_config_value(config: Dict[str, Any], title: str, key: str, indent: str = "\t") -> None:
    """Helper to print a single config value with proper formatting."""
    value = get_nested_value(config, key)
    log.info(f'{indent}{key.split(".")[-1]}: {" " * (25 - len(key.split(".")[-1]))}{value}')


def print_log_and_current_config(inference_config: Dict[str, Any], training_logged_config: Dict[str, Any]) -> None:
    # Original configuration overview
    log.info("ORIGINAL configuration")
    log.info(training_logged_config)

    # Original train dataset configuration
    log.info("ORIGINAL train dataset configuration")
    original_keys = [
        "data.pickle_path",
        "data.train_pickle_name",
        "data.val_pickle_name",
        "data.train_dataset_params.data_root_dir",
        "data.train_dataset_params.quantized_data_root_dir",
        "data.train_dataset_params.sequence_length",
        "data.train_dataset_params.subsampling_factor",
    ]

    for key in original_keys:
        print_config_value(training_logged_config, "ORIGINAL", key)

    # Current inference configuration
    log.info("CURRENT inference configuration")
    inference_keys = [
        "paths.pickle_path",
        "pickle_name",
        "paths.quantized_data_root_dir",
        "dataset_config.nb_context_frames",
        "dataset_config.nb_prediction_frames",
        "dataset_config.subsampling_factor",
    ]

    for key in inference_keys:
        print_config_value(inference_config, "CURRENT", key)


class GPUMemoryMonitor(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.max_memory: float = 0.0

    def on_predict_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Tuple[Tensor],
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if torch.cuda.is_available():
            memory: float = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            self.max_memory = max(self.max_memory, memory)
            trainer.logger.log_metrics({"gpu_memory_usage_gb": memory}, step=trainer.global_step)

    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if torch.cuda.is_available():
            trainer.logger.log_metrics({"max_gpu_memory_usage_gb": self.max_memory}, step=trainer.global_step)
        torch.cuda.reset_peak_memory_stats()
        self.max_memory = 0.0


class WorldModelPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, write_interval: str = "batch", max_queue_size: int = 50) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.max_queue_size = max_queue_size
        self.write_queue = []
        self.executor = ThreadPoolExecutor(max_workers=20)

    def write_on_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        prediction: Tuple[Tensor, List[str], int],
        batch_indices: List[int],
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        generated_data, image_paths, context_end_index = prediction

        for data_type, data in generated_data.items():

            if data_type not in ["visual_tokens", "visual_logits"]:
                continue

            output_subdir = self.output_dir / data_type
            self.queue_data(output_subdir, image_paths, context_end_index, data)

    def queue_data(self, output_dir: str, image_paths: List[str], context_end_index: int, data: Tensor) -> None:
        if data.device != "cpu":
            data = data.cpu()

        for b, sequence_image_paths in enumerate(image_paths):
            for t, image_path in enumerate(sequence_image_paths[context_end_index:]):

                output_path = (output_dir / image_path).with_suffix(".npy")
                self.write_queue.append((output_path, data[b, t]))

        if len(self.write_queue) >= self.max_queue_size:
            self.flush_queue()

    def flush_queue(self) -> None:
        futures = []
        for output_path, data in self.write_queue:
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            futures.append(self.executor.submit(np.save, output_path, data, allow_pickle=False))

        # Wait for all writes to complete
        for future in futures:
            future.result()

        self.write_queue.clear()

    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        self.flush_queue()  # Ensure all remaining data is written

    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self.executor.shutdown(wait=True)


class PredictionPathLogger(Callback):
    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.context_sequences = []
        self.generated_sequences = []

    def on_predict_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Tuple[Tensor],
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, image_paths, context_end_index = outputs

        for sequence_image_paths in image_paths:
            context_seq = sequence_image_paths[:context_end_index]
            generated_seq = sequence_image_paths[context_end_index:]

            self.context_sequences.append(context_seq)
            self.generated_sequences.append(generated_seq)

    def on_predict_epoch_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        # Gather data from all GPUs
        context_sequences = self.all_gather(self.context_sequences)
        generated_sequences = self.all_gather(self.generated_sequences)

        # Flatten the gathered lists
        context_sequences = [item for sublist in context_sequences for item in sublist]
        generated_sequences = [item for sublist in generated_sequences for item in sublist]

        # Write files only on the main process
        self.write_files(context_sequences, generated_sequences)

    def all_gather(self, data: Value) -> List[Value]:
        if torch.distributed.is_initialized():
            gathered_data = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_data, data)
            return gathered_data
        return [data]

    @rank_zero_only
    def write_files(self, context_sequences: List[str], generated_sequences: List[str]) -> None:
        # Write generated_frames.txt
        with open(self.output_dir / "generated_frames.txt", "w") as f:
            for seq in generated_sequences:
                for frame in seq:
                    f.write(f"{Path(frame).with_suffix('.npy')}\n")

        # Write context_sequences.txt
        with open(self.output_dir / "context_sequences.txt", "w") as f:
            for seq in context_sequences:
                f.write(" ; ".join(seq) + "\n")

        # Write generated_sequences.txt
        with open(self.output_dir / "generated_sequences.txt", "w") as f:
            for seq in generated_sequences:
                f.write(" ; ".join(seq) + "\n")


class WorldModelInference(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        sequence_adapter: nn.Module,
        action_tokenizer: nn.Module,
        sampler: Sampler,
        return_logits: bool,
        max_rolling_context_frames: Optional[int] = None,
    ) -> None:
        super().__init__()

        if max_rolling_context_frames is None:
            log.info(f"`max_rolling_context_frames` not set, using model's max context size: {network.nb_timesteps}")
            max_rolling_context_frames = network.nb_timesteps - 1

        self.action_tokenizer = action_tokenizer

        # necessary for auto moving to gpu device
        self.network = network
        self.sequence_adapter = sequence_adapter
        self.sampler = sampler

        self.inference_generator = ImageGenerator(
            self.network,
            self.sampler,
            self.sequence_adapter,
            max_rolling_context_frames,
        )

        self.gen_config = GenerationConfig(
            temperature=1.0,
            return_logits=return_logits,
            use_kv_cache=True,
            verbose=False,
        )

    def predict_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Tuple[Dict[str, Tensor], List[str], int]:
        action_tokens = self.action_tokenizer(**batch)

        burnin_visual_tokens = batch["visual_tokens"][:, : batch["context_end_index"]]
        burnin_action_tokens = action_tokens[:, : batch["context_end_index"]]

        future_action_tokens = action_tokens[:, batch["context_end_index"] :]

        generated_data = self.inference_generator.generate(
            burnin_visual_tokens, burnin_action_tokens, future_action_tokens, self.gen_config
        )

        return generated_data, batch["images_paths"], batch["context_end_index"]


def remove_prefix(state_dict: StateDict, prefix: str) -> Dict:
    result = {}
    for k, v in state_dict.items():
        tokens = k.split(".")
        if tokens[0] == prefix:
            tokens = tokens[1:]
            key = ".".join(tokens)
            result[key] = v
    return result


def load_model(
    checkpoint_path: str, model_config: DictConfig, device: str, inference_sha: str = None
) -> Tuple[nn.Module, ...]:
    checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))

    if checkpoint_data["git_sha"] != inference_sha and inference_sha is not None:
        log.warning("WARNING: checkpoint's git sha is different from the current one. Usually nothing to worry about.")
        log.warning(f"Checkpoint sha: {checkpoint_data['git_sha']}")

    network = instantiate(model_config.network)
    mup.set_base_shapes(network, base=model_config.mup_base_shapes)
    sequence_adapter = instantiate(model_config.sequence_adapter)
    action_tokenizer = instantiate(model_config.action_tokenizer)

    state_dict = remove_prefix(checkpoint_data["state_dict"], "network")
    network.load_state_dict(state_dict, strict=True)

    return network, sequence_adapter, action_tokenizer
