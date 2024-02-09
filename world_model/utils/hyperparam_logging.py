from typing import Any, Dict

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf
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

    hparams["data"] = config["data"]
    hparams["trainer"] = config["trainer"]

    hparams["callbacks"] = config.get("callbacks")
    hparams["name"] = config.get("name")
    hparams["ckpt_path"] = config.get("ckpt_path")
    hparams["seed"] = config.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
