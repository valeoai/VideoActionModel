from typing import Any, Dict, List, Tuple

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from world_model.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        config: A DictConfig configuration composed by Hydra.
        
    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    assert config.ckpt_path

    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(config.get("logger"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, logger=logger)

    object_dict = {
        "config": config,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=config.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=config.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(config: DictConfig) -> None:
    """Main entry point for evaluation.

    Args:
        config: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in config, print config tree, etc.)
    extras(config)

    evaluate(config)


if __name__ == "__main__":
    main()
