import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.amp
import torch.nn as nn
from cleanfid import resize as clean_resize
from einops import rearrange
from torch import Tensor
from torch.autograd import Function
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_info
from torchmetrics.utilities.imports import _SCIPY_AVAILABLE, _TORCH_FIDELITY_AVAILABLE
from torchvision.transforms import Normalize

if _TORCH_FIDELITY_AVAILABLE:
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
else:

    class FeatureExtractorInceptionV3(nn.Module):  # type: ignore
        pass

    __doctest_skip__ = ["FrechetInceptionDistance", "FID"]


if _SCIPY_AVAILABLE:
    import scipy

Kwargs = Dict[str, Any]


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.

    All credit to `Square Root of a Positive Definite Matrix`_
    """

    @staticmethod
    def forward(ctx: Function, input_data: Tensor) -> Tensor:
        # TODO: update whenever pytorch gets an matrix square root function
        # Issue: https://github.com/pytorch/pytorch/issues/9983
        m = input_data.detach().cpu().numpy().astype(np.float_)
        scipy_res, _ = scipy.linalg.sqrtm(m, disp=False)
        sqrtm = torch.from_numpy(scipy_res.real).to(input_data)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx: Function, grad_output: Tensor) -> Tensor:
        grad_input = None
        if ctx.needs_input_grad[0]:
            (sqrtm,) = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


class MultiInceptionMetrics(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(
        self,
        device: str,
        compute_manifold: bool = False,
        num_classes: int = 1000,
        num_inception_chunks: int = 10,
        manifold_k: int = 3,
        model: str = "inception",
        **kwargs: Kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.num_inception_chunks = num_inception_chunks
        self.manifold_k = manifold_k
        self.compute_manifold = compute_manifold

        if model == "inception":

            class NoTrainInceptionV3(FeatureExtractorInceptionV3):
                embed_dim: int = 2048

                def __init__(self, name: str, features_list: List[str], feature_extractor_weights_path: float = None) -> None:
                    super().__init__(name, features_list, feature_extractor_weights_path)

                @staticmethod
                def preprocess(image: Tensor) -> Tensor:
                    """convert from {(size, size), [-1, 1], float32} --> {(299, 299), [0, 255], int8}"""
                    # image = torch.nn.functional.interpolate(image, (299, 299), mode='bilinear', align_corners=False)
                    # image = (image + 1) / 2
                    # image = torch.clip(image * 255, 0, 255).to(torch.uint8)
                    # return image

                    # convert [-1, 1] to [0, 255]
                    image = (image + 1) / 2
                    image = torch.clip(image * 255, 0, 255)

                    # clean resize with cleanfid
                    l_resized_batch = torch.empty((len(image), 3, 299, 299), dtype=torch.uint8, device=image.device)
                    for idx in range(len(image)):
                        curr_img = image[idx]
                        img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                        img_resize = clean_resize.make_resizer("PIL", False, "bicubic", (299, 299))(img_np)
                        l_resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)), dtype=torch.uint8).unsqueeze(0)

                    # stack and convert to int8
                    return l_resized_batch

                def forward(self, x: Tensor) -> Tensor:
                    out, _ = super().forward(self.preprocess(x))
                    return out

            self.inception = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048", "logits_unbiased"])
            self.inception = self.inception.to(device)
            self.inception.eval()

        elif model == "dinov2":

            class _Dinov2(nn.Module):
                embed_dim: int = 1024

                def __init__(self) -> None:
                    super().__init__()
                    try:
                        base = os.path.expanduser(os.path.expandvars(os.environ.get("TORCH_HOME", "~/.cache")))
                        self.model = torch.hub.load(
                            os.path.join(base, "torch/hub/facebookresearch_dinov2_main/"), "dinov2_vitl14_reg", source="local"
                        )
                    except:  # noqa E722
                        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", force_reload=True)
                    self.model.to(device)
                    self.model.eval()
                    self.model.requires_grad_(False)
                    self.model.linear_head = nn.Identity()
                    self.scale = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

                def preprocess(self, images: Tensor) -> Tensor:
                    """convert from {(size, size), [-1, 1], float32} --> {(224, 224), [-1, 1], float32}"""
                    # select a size that is the clostest to the original size and divisible by 14
                    new_size = ((images.size(-2) // 14) * 14, (images.size(-1) // 14) * 14)
                    images = torch.nn.functional.interpolate(images, new_size, mode="bilinear")
                    return self.scale(images)

                def forward(self, x: Tensor) -> Tensor:
                    out = self.model(self.preprocess(x))
                    return out

            self.inception = _Dinov2().to(device)

        elif model == "clip":
            from transformers import CLIPModel

            class _Clip(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                    self.model.eval()
                    self.model.requires_grad_(False)

                @staticmethod
                def preprocess(images: Tensor) -> Tensor:
                    """convert from {(size, size), [-1, 1], float32} --> {(224, 224), [-1, 1], float32}"""
                    images = torch.nn.functional.interpolate(images, (224, 224))
                    return images

                def forward(self, x: Tensor) -> Tensor:
                    out = self.model.get_image_features(self.preprocess(x))
                    return out

            self.inception = _Clip().to(device)

        elif model == "i3d":
            # Use to compute the FVD
            # taking from
            # https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0/blob/810fa8c4bdb3a4c8eec9bd57375c29bde6fb46de/opensora/eval/fvd/styleganv/fvd.py
            class I3D(nn.Module):
                embed_dim: int = 400

                def __init__(self) -> None:
                    super().__init__()
                    self.model = self._load()
                    self.model.eval()
                    self.detector_kwargs = {
                        "rescale": False,
                        "resize": False,
                        "return_features": True,
                    }  # Return raw features before the softmax layer.

                @staticmethod
                def _load() -> nn.Module:
                    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
                    base = os.path.expanduser(os.path.expandvars(os.environ.get("TORCH_HOME", "~/.cache")))
                    filepath = os.path.join(base, "i3d_torchscript.pt")
                    print(filepath)
                    if not os.path.exists(filepath):
                        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
                        os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
                    i3d = torch.jit.load(filepath)
                    return i3d

                @staticmethod
                def preprocess(video: Tensor) -> Tensor:
                    """convert from {(size, size), [-1, 1], float32} --> {(224, 224), [-1, 1], float32}"""
                    # [-1, 1] -> [0, 255]
                    if video.ndim == 5:
                        # we add a dimension for compatibility with stochastic futures
                        video = video.unsqueeze(1)
                    b, s, t, *_ = video.size()
                    video = torch.nn.functional.interpolate(rearrange(video, "b s t c h w -> (b s t) c h w"), (224, 224))
                    # Convert to the correct format for I3D, i.e., --> B, C, T, H, W
                    video = rearrange(video, "(b s t) c h w -> (b s) c t h w", b=b, s=s, t=t)

                    # For FVD, the different futures will be consdiered as different samples
                    return video

                def forward(self, x: Tensor) -> Tensor:
                    # videos in [-1, 1] as torch tensor BTCHW
                    out = self.model(self.preprocess(x), **self.detector_kwargs)
                    return out

            self.inception = I3D().to(device)

        else:
            print("feature extractor does not exist")
            exit()

        self.add_state("real_count", torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("real_features_mean", torch.zeros(self.inception.embed_dim, device=device), dist_reduce_fx="sum")
        self.add_state(
            "real_features_cov",
            torch.zeros(self.inception.embed_dim, self.inception.embed_dim, device=device),
            dist_reduce_fx="sum",
        )
        self.add_state("fake_count", torch.tensor(0.0, device=device), dist_reduce_fx="sum")
        self.add_state("fake_features_mean", torch.zeros(self.inception.embed_dim, device=device), dist_reduce_fx="sum")
        self.add_state(
            "fake_features_cov",
            torch.zeros(self.inception.embed_dim, self.inception.embed_dim, device=device),
            dist_reduce_fx="sum",
        )

    def update(self, images: Tensor, image_type: str = "fake") -> None:
        # extract the features
        features = self.inception(images)
        features = features.view(features.size(0), -1)

        cov = torch.mm(features.t(), features)
        mean = features.sum(dim=0)
        count = features.size(0)

        if image_type == "real":
            self.real_count += count
            self.real_features_mean += mean
            self.real_features_cov += cov
        elif image_type == "fake":
            self.fake_count += count
            self.fake_features_mean += mean
            self.fake_features_cov += cov

    def _iterative_mean_cov(self, mean: Tensor, cov: Tensor, count: Tensor) -> Tensor:
        final_mean = mean / count
        final_cov = (cov - count * torch.outer(final_mean, final_mean)) / (count - 1)
        return final_mean, final_cov

    def fid(self) -> Tensor:
        real_mean, real_cov = self._iterative_mean_cov(self.real_features_mean, self.real_features_cov, self.real_count)
        fake_mean, fake_cov = self._iterative_mean_cov(self.fake_features_mean, self.fake_features_cov, self.fake_count)
        return self._compute_fid(real_mean, real_cov, fake_mean, fake_cov).item()

    def _compute_fid(self, mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, eps: float = 1e-6) -> Tensor:
        diff = mu1 - mu2

        covmean = sqrtm(sigma1.mm(sigma2))
        # Product might be almost singular
        if not torch.isfinite(covmean).all():
            rank_zero_info(f"FID calculation produces singular product; adding {eps} to diagonal of covariance estimates")
            offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
            covmean = sqrtm((sigma1 + offset).mm(sigma2 + offset))

        tr_covmean = torch.trace(covmean)
        return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

    def compute(self) -> Tensor:
        # Compute the actual score
        if self.real_count == 0 or self.fake_count == 0:
            return {"FID": float("nan")}
        return {"FID": self.fid()}
