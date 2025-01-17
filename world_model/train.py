import os
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from world_model.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(config: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        config: A DictConfig configuration composed by Hydra.

    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        lightning.seed_everything(config.seed, workers=True)

    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    if config.get("is_finetuning"):

        deepspeed_ckpt_dir = config.get("ckpt_path")
        pt_path = "checkpoint/mp_rank_00_model_states.pt"
        pt_path = os.path.join(deepspeed_ckpt_dir, pt_path)
        pt = torch.load(pt_path, map_location="cpu")
        pretrained_global_step = pt["global_step"]

        config.trainer.max_steps = pretrained_global_step + config.trainer.max_steps

        log.info(
            f"Finetuning | past global_step = {pretrained_global_step}, incrementing trainer max_steps tp {config.trainer.max_steps}"
        )
        log.info(
            f"Finetuning | scheduler.end_iter={config.scheduler.end_iter} ; scheduler.drop_iter={config.scheduler.drop_iter}"
        )

    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model, scheduler_conf=config.scheduler, _recursive_=False)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(config.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(config.get("logger"))

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "config": config,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if config.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.get("ckpt_path"))

        # Print path to best checkpoint
        if not config.trainer.get("fast_dev_run") and not (trainer.checkpoint_callback is None):
            log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

    train_metrics = trainer.callback_metrics

    if config.get("test") and not (trainer.checkpoint_callback is None):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        else:
            log.info(f"Best ckpt path: {ckpt_path}")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(config: DictConfig) -> Optional[float]:
    """Main entry point for training.

    Args:
        config: DictConfig configuration composed by Hydra.

    Returns
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. print config tree, etc.)
    extras(
        config,
        print_order=(
            "data",
            "model",
            "callbacks",
            "logger",
            "trainer",
            "paths",
        ),
    )

    # train the model
    metric_dict, _ = train(config)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=config.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
