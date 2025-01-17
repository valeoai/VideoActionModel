from typing import Any, Dict, Optional, Tuple

import git
import hydra
import mup
import torch
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from mup.optim import MuAdamW
from omegaconf import DictConfig
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from world_model.gpt2.prepare_token_sequence import prepare_AR_token_sequences


def remove_prefix(state_dict: Dict, prefix: str) -> Dict:
    result = dict()
    for k, v in state_dict.items():
        tokens = k.split(".")
        if tokens[0] == prefix:
            tokens = tokens[1:]
            key = ".".join(tokens)
            result[key] = v
    return result


Batch = Dict[str, torch.Tensor]
mupShapes = Dict[str, Tuple[int, ...]]


class NextTokenPredictor(LightningModule):
    """
    LightningModule docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer_conf: Optional[DictConfig] = None,
        scheduler_conf: Optional[DictConfig] = None,
        compile: bool = False,
        log_norm: bool = False,
        mup_base_shapes: mupShapes = None,
        statedict_ckpt_path: str = None,
        is_pretrained: bool = False,
    ) -> None:
        """
        Args:
            network: The configuration of the model to train.
            optimizer_conf: The optimizer to use for training.
            scheduler_conf: The learning rate scheduler to use for training.
            compile: Compile model for faster training with pytorch 2.0, registered with save_hyperparameters
            log_norm: log grad norm of model's parameters to loggers
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf
        self.network = hydra.utils.instantiate(network)
        self.mup_base_shapes = mup_base_shapes

        load_pretrained_network = statedict_ckpt_path is not None
        if load_pretrained_network:
            checkpoint_data = torch.load(statedict_ckpt_path, map_location=self.device)
            network_state_dict = remove_prefix(checkpoint_data["state_dict"], "network")
            self.network.load_pretrained_statedict(network_state_dict)

        if mup_base_shapes is not None:
            print("mup_base_shapes configured")
            rescale_params = not load_pretrained_network and not is_pretrained
            mup.set_base_shapes(self.network, mup_base_shapes, rescale_params=rescale_params)
            # re-initialize after set_base_shapes
            self.network.apply(self.network._init_weights)
        else:
            print("Network NOT mu-Parametrized")

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def on_before_optimizer_step(self, optimizer: Optional[Optimizer]) -> None:
        if self.hparams.log_norm:
            # Compute the 2-norm for each layer
            # If using mixed precision, the gradients are already unscaled here
            norms = grad_norm(self, norm_type=2)
            self.log_dict(norms)

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass

    def training_step(self, batch: Batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data.
            batch_idx: The index of the current batch.

        Returns:
         A tensor of losses between model predictions and targets.
        """

        input_data, target_data = prepare_AR_token_sequences(batch["visual_tokens"])

        logits_sequence = self.network(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")

        loss = self.cross_entropy_loss(logits_sequence, target_data["token_sequence"])

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
        input_data, target_data = prepare_AR_token_sequences(batch["visual_tokens"])

        logits_sequence = self.network(**input_data)
        logits_sequence = rearrange(logits_sequence, "b ... d -> b d ...")

        loss = self.cross_entropy_loss(logits_sequence, target_data["token_sequence"])

        # log losses at the end of epoch, rest is automatic
        self.log("val/loss_{dataloader_idx}", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        # return loss to apply backpropagation
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
            self.net = torch.compile(self.network, fullgraph=True)

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

        optimizer = MuAdamW(params=self.parameters(), **self.optimizer_conf)

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

        if self.mup_base_shapes is not None:
            checkpoint["mup_base_shapes"] = mup.shape._extract_shapes(self.mup_base_shapes)
