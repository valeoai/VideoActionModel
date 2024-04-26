from typing import Any, Dict
from omegaconf import OmegaConf
import git
import torch
from pathlib import Path

from lightning_utilities.core.rank_zero import rank_zero_only
from lightning.pytorch.core.saving import save_hparams_to_yaml
from world_model.utils.cmd_line_logging import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"config"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    config = OmegaConf.to_container(object_dict["config"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    
    hparams["paths"] = config["paths"]

    hparams["data"] = config["data"]
    hparams["trainer"] = config["trainer"]

    hparams["optimizer"] = config.get("optimizer")
    hparams["scheduler"] = config.get("scheduler")

    hparams["callbacks"] = config.get("callbacks")
    hparams["name"] = config.get("name")
    hparams["seed"] = config.get("seed")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        hparams["training_device"] = torch.cuda.get_device_name(0)
    else:
        hparams["training_device"] = "cpu"  # or handle the situation appropriately
    
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    hparams["git_sha"] = sha

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

    output_dir = Path(config["paths"]["output_dir"]) / config['name']
    save_hparams_to_yaml(output_dir / 'hparams.yaml', hparams, use_omegaconf=True)
    
    if not output_dir.exists():
        log.info(f'"{output_dir}" does not exists... creating it.')
        output_dir.mkdir(parents=True, exist_ok=True)    