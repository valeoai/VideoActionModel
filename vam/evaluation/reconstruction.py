from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric

Args = List[Any]
Kwargs = Dict[str, Any]


def remap_image_torch(image: Tensor) -> Tensor:
    """image should be between -1 and 1, convert it to [0, 1]"""
    image = (image + 1) / 2
    return image


class PixelsMetrics(Metric):

    def __init__(
        self,
        device: str,
        metrics: Union[str, List[str]] = None,
        data_range: float = 1,  # use data_range=2 to reproduce Llamagen results
        # https://github.com/FoundationVision/LlamaGen/blob/ce98ec41803a74a90ce68c40ababa9eaeffeb4ec/tokenizer/vae/reconstruction_vae_ddp.py#L164
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)

        if metrics is None:
            metrics = ["MSE", "PSNR", "SSIM"]
        if isinstance(metrics, str):
            metrics = [metrics]

        self.metrics = metrics
        self.add_state("count", torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        if "MSE" in metrics:
            self.add_state("mse_metrics", torch.tensor(0.0, device=device), dist_reduce_fx="sum")

        if "PSNR" in metrics:
            self.add_state("psnr_metrics", torch.tensor(0.0, device=device), dist_reduce_fx="sum")

        if "SSIM" in metrics:
            self.ssim = StructuralSimilarityIndexMeasure(data_range=data_range, reduction="none").to(device)
            self.add_state("ssim_metrics", torch.tensor(0.0, device=device), dist_reduce_fx="sum")

    @staticmethod
    def compute_mse(predictions: Tensor, targets: Tensor, reduction: str = "none") -> Tensor:
        # MSE (Mean Squared Error) function
        return F.mse_loss(predictions, targets, reduction=reduction)

    def compute_psnr(self, predictions: Tensor, targets: Tensor, max_pixel_value: float = 1.0) -> Tensor:
        # PSNR (Peak Signal-to-Noise Ratio) function
        mse = self.compute_mse(predictions, targets).mean(dim=(-3, -2, -1))
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse)).to(mse.device)
        return psnr

    def compute_ssim(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # SSIM (Structural Similarity Index Measure) function using torchmetrics
        return self.ssim(predictions, targets)

    def update(self, predictions: Tensor, targets: Tensor) -> None:
        predictions = remap_image_torch(predictions)
        targets = remap_image_torch(targets)

        self.count += torch.tensor(predictions.size(0), device=self.device)

        if "MSE" in self.metrics:
            batch_mse = self.compute_mse(predictions, targets)
            batch_mse = batch_mse.mean(dim=(1, 2, 3))
            self.mse_metrics += batch_mse.sum().to(self.device)

        if "PSNR" in self.metrics:
            batch_psnr = self.compute_psnr(predictions, targets)
            self.psnr_metrics += batch_psnr.sum().to(self.device)

        if "SSIM" in self.metrics:
            batch_ssim = self.compute_ssim(predictions, targets)
            self.ssim_metrics += batch_ssim.sum().to(self.device)

    def compute(self) -> Dict[str, float]:
        output_metrics = {}
        if "MSE" in self.metrics:
            output_metrics["MSE"] = self.mse_metrics / self.count

        if "PSNR" in self.metrics:
            output_metrics["PSNR"] = self.psnr_metrics / self.count

        if "SSIM" in self.metrics:
            output_metrics["SSIM"] = self.ssim_metrics / self.count

        return output_metrics


class VideoPixelMetrics(PixelsMetrics):

    def __init__(
        self,
        *args: Args,
        stochastic: bool = False,
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.stochastic = stochastic

    def update(self, video_predictions: Tensor, video_targets: Tensor) -> None:
        assert video_targets.ndim == 5, "Ground truth videos should have 5 dimensions"
        if self.stochastic:
            assert video_predictions.ndim == 6, "In stochastic mode video predictions should have 6 dimensions"
            video_targets = video_targets.unsqueeze(1).expand_as(video_predictions)
        else:
            assert video_predictions.ndim == 5, "Video predictions should have 5 dimensions"
            video_targets = video_targets.unsqueeze(1)
            video_predictions = video_predictions.unsqueeze(1)

        video_predictions = remap_image_torch(video_predictions)
        video_targets = remap_image_torch(video_targets)

        self.count += torch.tensor(video_predictions.size(0), device=self.device)

        if "MSE" in self.metrics:
            batch_mse = self.compute_mse(video_predictions, video_targets)
            batch_mse = batch_mse.mean(dim=(-4, -3, -2, -1))
            # take the lower along the possible futures dimension
            batch_mse = batch_mse.min(dim=1).values
            self.mse_metrics += batch_mse.sum().to(self.device)

        if "PSNR" in self.metrics:
            batch_psnr = self.compute_psnr(video_predictions, video_targets)
            # take the average along the time dimension
            # take the max along the possible futures dimension
            batch_psnr = batch_psnr.mean(-1).max(dim=1).values
            self.psnr_metrics += batch_psnr.sum().to(self.device)

        if "SSIM" in self.metrics:
            b, s, *_ = video_predictions.shape
            video_predictions = rearrange(video_predictions, "b s t c h w -> (b s t) c h w")
            video_targets = rearrange(video_targets, "b s t c h w -> (b s t) c h w")
            batch_ssim = self.compute_ssim(video_predictions, video_targets)
            batch_ssim = rearrange(batch_ssim, "(b s t) -> b s t", b=b, s=s)
            # take the average along the time dimension
            # take the max along the possible futures dimension
            batch_ssim = batch_ssim.mean(-1).max(dim=1).values
            self.ssim_metrics += batch_ssim.sum().to(self.device)


# if __name__ == '__main__':
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     tens1 = torch.rand(16, 2, 5, 3, 256, 256, device=device)
#     tens2 = torch.rand(16, 5, 3, 256, 256, device=device)

#     evaluator = VideoPixelMetrics('cuda', stochastic=True)
#     evaluator.update(tens1, tens2)
#     print(evaluator.compute())
