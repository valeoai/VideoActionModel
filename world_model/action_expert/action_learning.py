from typing import Any, Dict, Optional, Tuple

import git
import hydra
import lightning as L
import mup
import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from mup.optim import MuAdamW
from omegaconf import DictConfig
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from world_model.action_expert.vai0rbis import Vai0rbis

Batch = Dict[str, Tensor]
mupShapes = Dict[str, Tuple[int, ...]]


class ActionLearning(LightningModule):
    """
    LightningModule docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        vai0rbis_conf: DictConfig,
        optimizer_conf: Optional[DictConfig] = None,
        scheduler_conf: Optional[DictConfig] = None,
        flow_sampling: str = "uniform",
        flow_alpha: float = 1.5,
        flow_beta: float = 1,
        compile: bool = False,
        log_norm: bool = False,
        grad_logging: int = 0,
    ) -> None:
        """
        Args:
            vai0rbis_conf: The configuration for the Vai0rbis model.
            optimizer_conf: The optimizer to use for training.
            scheduler_conf: The learning rate scheduler to use for training.
            compile: Compile model for faster training with pytorch 2.0, registered with save_hyperparameters
            grad_logging: if > 0, logs histograms of the network's gradients every `grad_logging` steps
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.vai0rbis: Vai0rbis = hydra.utils.instantiate(vai0rbis_conf)
        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf
        self.grad_logging = grad_logging

        self.flow_sampling = flow_sampling
        if self.flow_sampling == "beta":
            self.flow_alpha = flow_alpha
            self.flow_beta = flow_beta
            self.flow_sig_min = self.vai0rbis.flow_sig_min
            self.flow_t_max = 1 - self.flow_sig_min
            self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

    def on_before_optimizer_step(self, optimizer: Optional[Optimizer]) -> None:
        if self.hparams.log_norm:
            # Compute the 2-norm for each layer
            # If using mixed precision, the gradients are already unscaled here
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

        if self.grad_logging <= 0:
            return

        for logger in self.loggers:
            if not isinstance(logger, L.pytorch.loggers.TensorBoardLogger):
                continue

            # inspect gradient information in tensorboard
            if self.global_step % self.grad_logging == 0:
                for k, v in self.vai0rbis.action_expert.named_parameters():
                    if v.grad is None:
                        print(f"{k} requires grad", v.requires_grad)
                        print(f"No grad for {k}")
                    else:
                        logger.experiment.add_histogram(tag=k, values=v.grad, global_step=self.global_step)

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def noise_schedule(self, batch: Batch) -> Tensor:
        """
        Adapted from:
        https://github.com/allenzren/open-pi-zero/blob/main/src/agent/train.py
        """
        batch_size, context_length, *_ = batch["visual_tokens"].size()
        bsz = batch_size * context_length

        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift

        t = rearrange(t, "(b c) -> b c", b=batch_size, c=context_length)
        return t.to(self.device, non_blocking=True)

    def training_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data.
            batch_idx: The index of the current batch.

        Returns:
         A tensor of losses between model predictions and targets.
        """

        diffusion_step = self.noise_schedule(batch)

        loss = self.vai0rbis(
            visual_tokens=batch["visual_tokens"],
            high_level_command=batch["high_level_command"],
            actions=batch["positions"],
            t=diffusion_step,
        )

        # log losses
        self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)

        # return loss to apply backpropagation
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def validation_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data.
            batch_idx: The index of the current batch.
        """
        diffusion_step = self.noise_schedule(batch)

        loss = self.vai0rbis(
            visual_tokens=batch["visual_tokens"],
            high_level_command=batch["high_level_command"],
            actions=batch["positions"],
            t=diffusion_step,
        )

        # log losses
        self.log(f"val/loss_{dataloader_idx}", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # return loss
        return loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends. Useful to log images for example."
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Args:
            stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            # torch.compile tries to compile all code in a model's forward() method.
            # Sections not compilable automatically cause "graph breaks",
            # splitting the code into optimized and unoptimized parts.
            # fullgraph=True to force an error if there is a graph break in the model,
            # calling for manual optimization of the code to get it compiled.
            self.net = torch.compile(self.vai0rbis, fullgraph=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        Returns:
            A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """

        if not self.optimizer_conf:
            return None

        print("MuAdamW configured with:", self.optimizer_conf)
        optimizer = MuAdamW(params=filter(lambda p: p.requires_grad is True, self.parameters()), **self.optimizer_conf)

        if not self.scheduler_conf:
            return {"optimizer": optimizer}

        scheduler = hydra.utils.instantiate(self.scheduler_conf, optimizer=optimizer)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "step",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            # "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            # "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "lr",
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def lr_scheduler_step(self, scheduler: _LRScheduler, metric: float) -> None:
        """
        Copy-pasting of Lightning code
        Manual override necessary for using custom LR schedulers otherwise it throws errors
        """
        if metric is None:
            scheduler.step()  # type: ignore[call-arg]
        else:
            scheduler.step(metric)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            sha = repo.head.object.hexsha

            checkpoint["git_sha"] = sha
        except git.exc.InvalidGitRepositoryError:
            checkpoint["git_sha"] = None

        # save class name of the model in the checkpoint
        checkpoint["model_class_path"] = self.__module__ + "." + self.__class__.__qualname__

        if self.vai0rbis.gpt_mup_base_shapes is not None:
            checkpoint["gpt_mup_base_shapes"] = mup.shape._extract_shapes(self.vai0rbis.gpt_mup_base_shapes)
        if self.vai0rbis.action_mup_base_shapes is not None:
            checkpoint["action_mup_base_shapess"] = mup.shape._extract_shapes(self.vai0rbis.action_mup_base_shapes)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/experiment/action_learning.yaml")
    data_config = OmegaConf.load("configs/data/ego_trajectory_dataset.yaml")

    vai0rbis_conf = config.model.vai0rbis_conf
    optimizer_conf = config.model.optimizer_conf

    vai0rbis_conf.finetuning_timesteps = 8
    vai0rbis_conf.action_config.action_horizon = 6

    model = ActionLearning(vai0rbis_conf, optimizer_conf)
    model.to("cuda")

    data_config.nuscenes_tokens_rootdir = config.data.nuscenes_tokens_rootdir
    data_config.nuscenes_train_pickle_path = config.data.nuscenes_train_pickle_path
    data_config.nuscenes_val_pickle_path = config.data.nuscenes_val_pickle_path

    dm = hydra.utils.instantiate(data_config)
    dm = dm.setup("fit")
    train_loader = dm.train_dataloader()

    batch = next(iter(train_loader))

    def _to(v: Tensor | list) -> Tensor | list:
        if isinstance(v, Tensor):
            return v.to(model.device)

        return v

    batch = {k: _to(v) for k, v in batch.items()}

    model.training_step(batch, 0)
