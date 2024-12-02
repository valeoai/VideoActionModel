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
try:
    import git
except ImportError:
    git = None
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import mup
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import BasePredictionWriter, TQDMProgressBar, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from world_model.dataloader.components.scene_tokenized_sequence_nuplan import SceneBasedNuplanDataset
from world_model.dataloader.tokenized_sequence_nuplan import custom_collate
from world_model.utils.generation import ImageGenerator, GenerationConfig
from world_model.utils import RankedLogger, extras, instantiate_samplers

log = RankedLogger(__name__, rank_zero_only=True)


def worker_rnd_init(x):
    np.random.seed(42 + x)


def get_nested_value(config, key_path):
    """Helper to safely get nested dictionary values using dot notation."""
    try:
        value = config
        for key in key_path.split('.'):
            value = value[key] if isinstance(value, dict) else getattr(value, key)
        return value
    except (KeyError, AttributeError):
        return "NOT FOUND"


def print_config_value(config, title, key, indent='\t'):
    """Helper to print a single config value with proper formatting."""
    value = get_nested_value(config, key)
    log.info(f'{indent}{key.split(".")[-1]}: {" " * (25 - len(key.split(".")[-1]))}{value}')


def print_log_and_current_config(inference_config, training_logged_config):
    # Original configuration overview
    log.info('ORIGINAL configuration')
    log.info(training_logged_config)

    # Original train dataset configuration
    log.info('ORIGINAL train dataset configuration')
    original_keys = [
        'data.pickle_path',
        'data.train_pickle_name',
        'data.val_pickle_name',
        'data.train_dataset_params.data_root_dir',
        'data.train_dataset_params.quantized_data_root_dir',
        'data.train_dataset_params.sequence_length',
        'data.train_dataset_params.subsampling_factor'
    ]

    for key in original_keys:
        print_config_value(training_logged_config, 'ORIGINAL', key)

    # Current inference configuration
    log.info('CURRENT inference configuration')
    inference_keys = [
        'paths.pickle_path',
        'pickle_name',
        'paths.quantized_data_root_dir',
        'dataset_config.nb_context_frames',
        'dataset_config.nb_prediction_frames',
        'dataset_config.subsampling_factor'
    ]

    for key in inference_keys:
        print_config_value(inference_config, 'CURRENT', key)


class GPUMemoryMonitor(Callback):
    def __init__(self):
        super().__init__()
        self.max_memory = 0

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if torch.cuda.is_available():
            memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert to GB
            self.max_memory = max(self.max_memory, memory)
            trainer.logger.log_metrics({"gpu_memory_usage_gb": memory}, step=trainer.global_step)

    def on_predict_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            trainer.logger.log_metrics({"max_gpu_memory_usage_gb": self.max_memory}, step=trainer.global_step)
        torch.cuda.reset_peak_memory_stats()
        self.max_memory = 0


class WorldModelPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir: Path, write_interval: str = "batch", max_queue_size: int = 50):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.max_queue_size = max_queue_size
        self.write_queue = []
        self.executor = ThreadPoolExecutor(max_workers=20)

    def write_on_batch_end(
        self, trainer, pl_module, prediction: Any, batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        generated_data, image_paths, context_end_index = prediction

        for data_type, data in generated_data.items():

            if data_type not in ['visual_tokens', 'visual_logits']:
                continue

            output_subdir = self.output_dir / data_type
            self.queue_data(output_subdir, image_paths, context_end_index, data)

    def queue_data(self, output_dir, image_paths, context_end_index, data):
        if data.device != "cpu":
            data = data.cpu()

        for b, sequence_image_paths in enumerate(image_paths):
            for t, image_path in enumerate(sequence_image_paths[context_end_index:]):

                output_path = (output_dir / image_path).with_suffix('.npy')
                self.write_queue.append((output_path, data[b, t]))

        if len(self.write_queue) >= self.max_queue_size:
            self.flush_queue()

    def flush_queue(self):
        futures = []
        for output_path, data in self.write_queue:
            if not output_path.parent.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
            futures.append(self.executor.submit(np.save, output_path, data, allow_pickle=False))

        # Wait for all writes to complete
        for future in futures:
            future.result()

        self.write_queue.clear()

    def on_predict_epoch_end(self, trainer, pl_module):
        self.flush_queue()  # Ensure all remaining data is written

    def teardown(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self.executor.shutdown(wait=True)


class PredictionPathLogger(Callback):
    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = output_dir
        self.context_sequences = []
        self.generated_sequences = []

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        _, image_paths, context_end_index = outputs

        for sequence_image_paths in image_paths:
            context_seq = sequence_image_paths[:context_end_index]
            generated_seq = sequence_image_paths[context_end_index:]

            self.context_sequences.append(context_seq)
            self.generated_sequences.append(generated_seq)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Gather data from all GPUs
        context_sequences = self.all_gather(self.context_sequences)
        generated_sequences = self.all_gather(self.generated_sequences)

        # Flatten the gathered lists
        context_sequences = [item for sublist in context_sequences for item in sublist]
        generated_sequences = [item for sublist in generated_sequences for item in sublist]

        # Write files only on the main process
        self.write_files(context_sequences, generated_sequences)

    def all_gather(self, data):
        if torch.distributed.is_initialized():
            gathered_data = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered_data, data)
            return gathered_data
        return [data]

    @rank_zero_only
    def write_files(self, context_sequences, generated_sequences):
        # Write generated_frames.txt
        with open(self.output_dir / 'generated_frames.txt', 'w') as f:
            for seq in generated_sequences:
                for frame in seq:
                    f.write(f"{Path(frame).with_suffix('.npy')}\n")

        # Write context_sequences.txt
        with open(self.output_dir / 'context_sequences.txt', 'w') as f:
            for seq in context_sequences:
                f.write(" ; ".join(seq) + "\n")

        # Write generated_sequences.txt
        with open(self.output_dir / 'generated_sequences.txt', 'w') as f:
            for seq in generated_sequences:
                f.write(" ; ".join(seq) + "\n")


class WorldModelInference(L.LightningModule):
    def __init__(
        self,
        network,
        sequence_adapter,
        action_tokenizer,
        sampler,
        return_logits,
        max_rolling_context_frames=None,
        verbose=False,
    ):
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
            verbose=verbose,
        )

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        action_tokens = self.action_tokenizer(**batch)

        burnin_visual_tokens = batch['visual_tokens'][:, :batch['context_end_index']]
        burnin_action_tokens = action_tokens[:, :batch['context_end_index']]

        future_action_tokens = action_tokens[:, batch['context_end_index']:]

        generated_data = self.inference_generator.generate(
            burnin_visual_tokens,
            burnin_action_tokens,
            future_action_tokens,
            self.gen_config,
        )

        return generated_data, batch['images_paths'], batch['context_end_index']


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()
    for k, v in state_dict.items():
        tokens = k.split('.')
        if tokens[0] == prefix:
            tokens = tokens[1:]
            key = '.'.join(tokens)
            result[key] = v
    return result


def load_model(
    checkpoint_path,
    model_config,
    device,
    inference_sha=None,
):
    if isinstance(checkpoint_path, str):
        checkpoint_data = torch.load(checkpoint_path, map_location=torch.device(device))
    elif isinstance(checkpoint_path, dict):
        checkpoint_data = checkpoint_path
    else:
        raise ValueError(f"Invalid checkpoint path type: {type(checkpoint_path)}")

    if checkpoint_data["git_sha"] != inference_sha and inference_sha is not None:
        log.warning("WARNING: checkpoint's git sha is different from the current one. Usually nothing to worry about.")
        log.warning(f"Checkpoint sha: {checkpoint_data['git_sha']}")

    network = instantiate(model_config.network)
    mup.set_base_shapes(network, base=model_config.mup_base_shapes)
    sequence_adapter = instantiate(model_config.sequence_adapter)
    action_tokenizer = instantiate(model_config.action_tokenizer)

    state_dict = remove_prefix(checkpoint_data['state_dict'], 'network')
    network.load_state_dict(state_dict, strict=True)

    return network, sequence_adapter, action_tokenizer


def load_data(pickle_path, dataset_config, dataloader_config):
    with open(pickle_path, 'rb') as f:
        pickle_data = pickle.load(f)

    dataset = SceneBasedNuplanDataset(
        pickle_data=pickle_data,
        **dataset_config
    )

    dataloader = DataLoader(
        dataset,
        worker_init_fn=worker_rnd_init,
        collate_fn=custom_collate,
        **dataloader_config
    )

    return dataloader


def infer(inference_config: DictConfig) -> None:
    checkpoint_name = Path(inference_config.checkpoint_name)

    model_log_dir = Path(inference_config.model_log_dir)
    checkpoint_path = model_log_dir / 'checkpoints' / checkpoint_name
    assert checkpoint_path.exists(), str(checkpoint_path)

    base_output_dir = Path(inference_config.output_dir)
    base_output_dir = (
        base_output_dir /
        model_log_dir.name /
        'inference' /
        checkpoint_name.with_suffix('')
    )

    log.info("Initializing...")
    if not base_output_dir.exists():
        log.info(f'"{base_output_dir}" does not exists... creating it.')
        base_output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    repo = git.Repo(search_parent_directories=True)
    inference_sha = repo.head.object.hexsha

    log.info("Loading the model config from logged data...")
    with open(model_log_dir / 'hparams.yaml', 'r') as file:
        training_logged_config = yaml.safe_load(file)
    training_logged_config = OmegaConf.create(training_logged_config)
    model_config = training_logged_config.model

    print_log_and_current_config(inference_config, training_logged_config)

    log.info("Instantiating the model...")
    network, sequence_adapter, action_tokenizer = load_model(
        checkpoint_path,
        model_config,
        device,
        inference_sha
    )

    log.info("Instantiating the dataset...")
    pickle_path = Path(inference_config.paths.pickle_path) / inference_config.pickle_name
    dataloader = load_data(pickle_path, inference_config.dataset_config, inference_config.dataloader_config)

    log.info("Instantiating samplers...")
    samplers = instantiate_samplers(inference_config.samplers)

    for sampler in samplers:
        for sampling_idx in range(inference_config.nb_samplings):
            output_dir = base_output_dir / f'{sampler}_sampling' / f'generation_{sampling_idx}'

            model = WorldModelInference(
                network,
                sequence_adapter,
                action_tokenizer,
                sampler,
                return_logits=(sampling_idx == 0 and inference_config.save_logits),
                max_rolling_context_frames=inference_config.max_rolling_context_frames
            )

            prediction_writer = WorldModelPredictionWriter(output_dir, max_queue_size=inference_config.max_queue_size)
            prediction_path_logger = PredictionPathLogger(base_output_dir)
            progress_bar = TQDMProgressBar(refresh_rate=5)
            gpu_memory_monitor = GPUMemoryMonitor()

            tensorboard_logger = TensorBoardLogger(
                save_dir=inference_config.paths.output_dir,
                name=f"{inference_config.name}/tensorboard/",
                log_graph=False,
                prefix=""
            )

            trainer = L.Trainer(
                devices=inference_config.trainer.devices,
                num_nodes=inference_config.trainer.num_nodes,
                strategy=inference_config.trainer.strategy,
                precision=inference_config.trainer.precision,
                accelerator=inference_config.trainer.accelerator,
                callbacks=[prediction_writer, prediction_path_logger, progress_bar, gpu_memory_monitor],
                logger=tensorboard_logger,
                enable_progress_bar=True
            )

            trainer.predict(model, dataloader)

    log.info("Processing complete!")

    OmegaConf.update(inference_config, 'git_sha', inference_sha, force_add=True)
    meta_config = DictConfig({
        'inference_config': inference_config,
        'training_config': training_logged_config
    })

    metada_path = base_output_dir / 'generated_tokens_metadata.yaml'
    OmegaConf.save(meta_config, metada_path)

    log.info(f"Metada saved at: {metada_path}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(config: DictConfig) -> None:
    extras(config)
    infer(config)


if __name__ == "__main__":
    main()
