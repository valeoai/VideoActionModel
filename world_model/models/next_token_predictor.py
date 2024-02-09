from typing import Any, Dict, List, Tuple
import hydra
from omegaconf import DictConfig
from typing import Optional
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import lightning
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from torchmetrics import MeanMetric


class NextTokenPredictor(LightningModule):
    """
    LightningModule docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        action_quantizer: DictConfig,
        sequence_adapter: DictConfig,
        optimizer_conf: Optional[DictConfig] = None,
        scheduler_conf: Optional[DictConfig] = None,
        compile: bool = False,
    ) -> None:
        """
        Args:
            network: The configuration of the model to train.
            action_quantizer: The configuration for a callable that takes as input the ego motion data and produce discrete tokens.
            sequence_adapter: The configuration of an adapter the produce a unified sequence of tokens from visual and action tokens.
            optimizer_conf: The optimizer to use for training.
            scheduler_conf: The learning rate scheduler to use for training.
            compile: Compile model for faster training with pytorch 2.0, registered with save_hyperparameters
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        
        self.optimizer_conf = optimizer_conf
        self.scheduler_conf = scheduler_conf
        self.network = hydra.utils.instantiate(network)
        self.action_quantizer = hydra.utils.instantiate(action_quantizer)
        self.sequence_adapter = hydra.utils.instantiate(sequence_adapter)
        
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        # for averaging loss across batches
        self.mean_val_loss = MeanMetric()
    
    def on_before_optimizer_step(self, optimizer):
        """
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)
        """
        pass
    
    def create_inputs_and_target(self, batch):
        
        action_tokens = self.action_quantizer(**batch)
        
        tokens_sequence, visual_tokens_mask = self.sequence_adapter(batch['codebook_indices'], action_tokens)
        
        # Create target_tokens by taking all but the first token (shifting by one)
        input_tokens = tokens_sequence[:, :-1]
        target_tokens = tokens_sequence[:, 1:]
        target_visual_tokens_mask = visual_tokens_mask[:, 1:]
        
        return {
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'target_visual_tokens_mask': target_visual_tokens_mask
        }
        
    def model_step(self, batch: Any) -> torch.Tensor:
        """Perform a single model step on a batch of data."""
        
        sequence_data = self.create_inputs_and_target(batch)
                
        logits_sequence = self.network(sequence_data['input_tokens'])
        
        visual_logits = logits_sequence[sequence_data['target_visual_tokens_mask']]
        visual_target_tokens = sequence_data['target_tokens'][sequence_data['target_visual_tokens_mask']]
        
        loss = self.cross_entropy_loss (visual_logits, visual_target_tokens)
        
        return {
            'loss': loss
        }
        
    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when training begins."""
        pass
        
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        Args:
            batch: A batch of data.
            batch_idx: The index of the current batch.
        
        Returns:
         A tensor of losses between model predictions and targets.
        """
        
        loss = self.model_step(batch)['loss']

        # log losses
        self.log("train/loss", loss, on_step=True, on_epoch=False, logger=True, prog_bar=True)

        # return loss to apply backpropagation
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass
        
    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.mean_val_loss.reset()
        
    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        Args:
            batch: A batch of data.
            batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)['loss']

        # update metric aggregator
        self.mean_val_loss(loss)
        
        # log metric object at each epoch, metric is automatically reset by lightning after each epoch
        self.log("val/loss", self.mean_val_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)

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
            self.net = torch.compile(self.network)

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

        optimizer = hydra.utils.instantiate(
            self.optimizer_conf, params=self.parameters()
        )

        if not self.scheduler_conf:
            return {"optimizer": optimizer}

        scheduler = hydra.utils.instantiate(
            self.scheduler_conf, optimizer=optimizer
        )

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": scheduler,
            
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            
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

    def on_save_checkpoint(self, checkpoint):
        # save class name of the model in the checkpoint
        checkpoint["model_class_path"] = (
            self.__module__ + "." + self.__class__.__qualname__
        )
