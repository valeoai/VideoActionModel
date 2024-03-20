from typing import List

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from world_model.utils.cmd_line_logging import RankedLogger
from world_model.utils.generation import Sampler

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_config: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_config: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_config:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_config, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_config.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_config: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_config: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    logger: List[Logger] = []

    if not logger_config:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_config, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_config.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger

def instantiate_samplers(samplers_config: DictConfig) -> List[Sampler]:
    """Instantiates samplers from config.

    :param samplers_config: A DictConfig object containing sampler configurations.
    :return: A list of instantiated samplers.
    """
    samplers: List[Sampler] = []

    if not samplers_config:
        log.warning("No logger configs found! Skipping...")
        return samplers

    if not isinstance(samplers_config, DictConfig):
        raise TypeError("Sampler config must be a DictConfig!")

    for _, sampler_config in samplers_config.items():
        if isinstance(sampler_config, DictConfig) and "_target_" in sampler_config:
            log.info(f"Instantiating sampler <{sampler_config._target_}>")
            sampler = hydra.utils.instantiate(sampler_config)
            if not isinstance(sampler, Sampler):
                raise TypeError(f"Sampler class <{sampler_config._target_}> does not inherit from base `Sampler` class!")
            samplers.append(sampler)

    return samplers
