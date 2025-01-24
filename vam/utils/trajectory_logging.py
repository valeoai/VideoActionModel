from typing import Any, Dict

import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, Trainer
from lightning.pytorch.utilities import move_data_to_device
from torch.utils.data import DataLoader, default_collate

from vam.action_expert import ActionLearning
from vam.utils.cmd_line_logging import RankedLogger

Batch = Dict[str, Any]

log = RankedLogger(__name__, rank_zero_only=True)


class TrajectoryLoggingCallback(Callback):

    def __init__(self, train_log_every_n_step: int, num_trajs_to_log: int) -> None:
        self.train_log_every_n_step = train_log_every_n_step
        self.num_trajs_to_log = num_trajs_to_log

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: ActionLearning, *args, **kwargs) -> None:
        # only log on first GPU
        if trainer.global_rank != 0:
            return

        if isinstance(trainer.val_dataloaders, list):
            log.warning(
                "there are more than one val dataloader, logic for handling multiple val dataset no implemented, "
                "only logging val_dataloaders[0]"
            )
            dataloader = trainer.val_dataloaders[0]
        else:
            dataloader = trainer.val_dataloaders

        batch_to_log = self.get_batch_to_log(dataloader)
        batch_to_log = move_data_to_device(batch_to_log, pl_module.device)

        self.common_logging("val", batch_to_log, trainer, pl_module, trainer.current_epoch)

    def on_train_batch_end(self, trainer: Trainer, pl_module: ActionLearning, *args, **kwargs) -> None:
        if trainer.global_rank != 0:
            return

        if self.train_log_every_n_step == 0:
            return

        if trainer.global_step % self.train_log_every_n_step != 0 or trainer.global_step == 0:
            return

        batch_to_log = self.get_batch_to_log(trainer.train_dataloader)
        batch_to_log = move_data_to_device(batch_to_log, pl_module.device)

        pl_module.vam.eval()
        self.common_logging("train", batch_to_log, trainer, pl_module, trainer.global_step)
        pl_module.vam.train()

    def get_batch_to_log(self, dataloader: DataLoader) -> Batch:
        len_dataset = len(dataloader.dataset)

        if len_dataset >= self.num_trajs_to_log:
            idxs_to_sample = np.arange(0, len_dataset - 1, len_dataset // self.num_trajs_to_log)
        else:
            idxs_to_sample = [0]

        elems_to_log = []
        for idx in idxs_to_sample:
            elems_to_log.append(dataloader.dataset[idx])

        batch_to_log = default_collate(elems_to_log)

        return batch_to_log

    def plot_trajectory_comparison(self, pred_traj: np.ndarray, gt_traj: np.ndarray) -> plt.Figure:
        """Create a comparison plot of predicted vs ground truth trajectory"""
        # Calculate data ranges to determine appropriate figure size
        x_range = max(np.ptp(pred_traj[:, 0]), np.ptp(gt_traj[:, 0]))
        y_range = max(np.ptp(pred_traj[:, 1]), np.ptp(gt_traj[:, 1]))
        aspect_ratio = y_range / x_range if x_range != 0 else 1

        # Adjust figure size based on aspect ratio while maintaining reasonable dimensions
        width = 6
        height = max(min(width * aspect_ratio, 10), 4)  # Limit height between 4 and 10

        fig, ax = plt.subplots(figsize=(width, height))

        # Plot trajectories
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], "b-", label="GT", alpha=0.7)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], "r--", label="Reconstruction", alpha=0.7)

        ax.set_aspect("equal")
        ax.grid(True)

        # Place legend below the plot
        ax.legend(bbox_to_anchor=(0.5, -0.15), loc="upper center", ncol=2)

        # Adjust layout to prevent legend cutoff
        plt.tight_layout()

        return fig

    def common_logging(
        self, phase: str, batch_to_log: Batch, trainer: Trainer, pl_module: ActionLearning, log_step: int = 0
    ) -> None:

        for logger in trainer.loggers:
            if not isinstance(logger, lightning.pytorch.loggers.TensorBoardLogger):
                continue

            with torch.no_grad():
                recon = pl_module.vam.forward_inference(
                    batch_to_log["visual_tokens"],
                    batch_to_log["high_level_command"][:, -1:],
                    dtype=batch_to_log["positions"].dtype,
                )

                pl_module.vam.joint_model.kv_cache = None

            for i in range(self.num_trajs_to_log):
                traj_title = f"{phase}/traj_{i}"

                plt_figure = self.plot_trajectory_comparison(
                    recon[i, 0].cpu().numpy(), batch_to_log["positions"][i, -1].cpu().numpy()
                )

                logger.experiment.add_figure(traj_title, plt_figure, global_step=log_step)
                plt.close("all")
