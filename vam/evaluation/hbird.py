import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.amp
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(iterator: Iterator, *_, **__) -> Iterator:
        return iterator


from vam.evaluation.depth import DepthMetrics
from vam.evaluation.miou import fast_cm_torch, per_class_iou_torch

Info = Dict[str, Any]
Kwargs = Dict[str, Any]
FeaturesFunction = Callable[[Tensor, bool], Tensor]


def batched_bincount(x: Tensor, max_value: int, dim: int = -1) -> Tensor:
    # adapted from
    # https://discuss.pytorch.org/t/batched-bincount/72819/3
    shape = x.shape[:-1] + (max_value,)
    target = torch.zeros(*shape, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target


def histogram_binning(x: Tensor, bins: int) -> Tensor:
    min_val, max_val = x.min(), x.max()
    x = (x - min_val) / (max_val - min_val)
    x = (x * bins).long()
    return x


class HbirdEvaluation:
    def __init__(
        self,
        *,
        ftr_extr_fn: FeaturesFunction,
        model_info: Info,
        dataset_info: Info,
        train_loader: DataLoader,
        num_neighbour: int,
        augmentation_epoch: int,
        device: str = "cuda",
        dtype: str = "bf16",
        evaluation_task: str = "segmentation",
        num_bins: int = 255,
        is_distributed: bool = False,
        nn_params: Optional[Kwargs] = None,
        memory_size: Optional[int] = None,
        f_mem_p: Optional[str] = None,
        l_mem_p: Optional[str] = None,
    ) -> None:
        assert evaluation_task in ["segmentation", "depth"], "Evaluation task should be either segmentation or depth"
        if nn_params is None:
            nn_params = {}
        self.ftr_extr_fn = ftr_extr_fn
        self.device = device
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
        self.is_distributed = is_distributed
        self.augmentation_epoch = augmentation_epoch
        self.memory_size = memory_size
        self.num_neighbour = num_neighbour
        self.num_bins = num_bins
        self.evaluation_task = evaluation_task
        self.num_classes = dataset_info["num_classes"]
        eval_spatial_resolution = model_info["eval_spatial_resolution"]
        self.num_sampled_features = None
        self.window_size = dataset_info["window_size"]
        self.f_mem_p = f_mem_p
        self.l_mem_p = l_mem_p

        if self.evaluation_task == "segmentation":
            self.gt_key = "mask"
        elif self.evaluation_task == "depth":
            self.gt_key = "depth"

        if not self.load_memory():
            dataset_size = dataset_info["dataset_size"]
            window_size = dataset_info["window_size"]
            patch_size = model_info["patch_size"]
            d_model = model_info["d_model"]

            if self.memory_size is not None:
                assert (
                    self.memory_size % (dataset_size * self.augmentation_epoch * window_size) == 0
                ), "Memory size should be multiple of dataset size"
                self.num_sampled_features = self.memory_size // (dataset_size * self.augmentation_epoch * window_size)
                print("Number of sampled features: ", self.num_sampled_features)
                ## create memory of specific size
                self.feature_memory = torch.zeros((self.memory_size, d_model))
                if self.evaluation_task == "segmentation":
                    self.label_memory = torch.zeros((self.memory_size, self.num_classes))
                elif self.evaluation_task == "depth":
                    self.label_memory = torch.zeros((self.memory_size, patch_size * patch_size))
            self.create_memory(train_loader, num_classes=self.num_classes, eval_spatial_resolution=eval_spatial_resolution)
            self.save_memory()

        self.feature_memory = self.feature_memory.to(self.device)
        self.label_memory = self.label_memory.to(self.device)
        norm = torch.norm(self.feature_memory, dim=1)
        # check if some zeros or nans
        if torch.any(norm == 0) or torch.any(torch.isnan(norm)):
            raise ValueError("Some features have norm 0 or nan")
        self.create_NN(self.num_neighbour, **nn_params)

    def create_NN(
        self,
        num_neighbour: int = 30,
        num_leaves_to_search: Optional[int] = None,
        num_leaves: Optional[int] = None,
        num_reordering_candidates: Optional[int] = None,
        distance_measure: str = "dot_product",
        anisotropic_quantization_threshold: float = 0.2,
    ) -> None:
        """
        following advices from:
        https://github.com/google-research/google-research/blob/master/scann/docs/algorithms.md
        """
        try:
            import scann
        except ImportError:
            print(
                "Scann is not installed. It is used only for HBird eval. "
                "Please install it using 'pip install scann' if you want to run this evaluation."
            )

        if num_reordering_candidates is None:
            num_reordering_candidates = num_neighbour * 10
            print(
                "Number of reordering candidates is not provided, setting it to 10 times the number of neighbours, i.e. ",
                num_reordering_candidates,
            )
        if num_leaves is None:
            num_leaves = int(np.sqrt(self.feature_memory.size(0)))
            print("Number of leaves is not provided, setting it to ", num_leaves)
        if num_leaves_to_search is None:
            num_leaves_to_search = num_leaves // 10
            print("Number of leaves to search is not provided, setting it to ", num_leaves_to_search)

        self.NN_algorithm = (
            scann.scann_ops_pybind.builder(
                self.feature_memory.detach().cpu().numpy(),
                num_neighbour,
                distance_measure,
            )
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=self.feature_memory.size(0),
            )
            .score_ah(2, anisotropic_quantization_threshold=anisotropic_quantization_threshold)
            .reorder(
                num_reordering_candidates,
            )
            .build()
        )

    @torch.no_grad()
    def create_memory(self, train_loader: DataLoader, num_classes: int, eval_spatial_resolution: Tuple[int, int]) -> None:
        feature_memory = []
        label_memory = []
        idx = 0
        for _ in range(self.augmentation_epoch):
            for _, batch in enumerate(tqdm(train_loader, desc="Memory Creation loop")):
                x, y = batch["image"], batch[self.gt_key]

                if x.ndim == 5:
                    # the model should handle the temporal dimension
                    y = rearrange(y, "b t c h w -> (b t) c h w")

                x = x.to(self.device)
                y = y.to(self.device)
                with torch.amp.autocast(self.device, dtype=self.dtype):
                    features = self.ftr_extr_fn(x, inference=False)
                features = features.float()
                patch_size = x.shape[-1] // eval_spatial_resolution[-1]
                if self.evaluation_task == "segmentation":
                    y = y.long()
                    y[y == 255] = 0
                    patchified_gts = self.patchify_gt(
                        y, patch_size
                    )  # (bs, spatial_resolution, spatial_resolution, c*patch_size*patch_size)
                    one_hot_patch_gt = F.one_hot(patchified_gts, num_classes=num_classes).float()
                    label = one_hot_patch_gt.mean(dim=3)
                elif self.evaluation_task == "depth":
                    patchified_gts = self.patchify_depth(y, patch_size)
                    label = patchified_gts

                if self.memory_size is None:
                    # Memory Size is unbounded so we store all the features
                    normalized_features = features / torch.norm(features, dim=-1, keepdim=True)

                    normalized_features = normalized_features.flatten(0, 1)
                    label = label.flatten(0, 2)
                    feature_memory.append(normalized_features.detach().cpu())
                    label_memory.append(label.detach().cpu())
                else:
                    # Memory Size is bounded so we need to select/sample some features only
                    sampled_features, sampled_indices = self.sample_features_batch(features, patchified_gts)
                    normalized_sampled_features = sampled_features / torch.norm(sampled_features, dim=-1, keepdim=True)
                    label = label.flatten(1, 2)
                    ## select the labels of the sampled features
                    sampled_indices = sampled_indices.to(self.device)
                    ## repeat the label for each sampled feature
                    label_hat = label.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, label.shape[-1]))

                    # label_hat = label.gather(1, sampled_indices)
                    normalized_sampled_features = normalized_sampled_features.flatten(0, 1)
                    label_hat = label_hat.flatten(0, 1)
                    self.feature_memory[idx : idx + normalized_sampled_features.size(0)] = (
                        normalized_sampled_features.detach().cpu()
                    )
                    self.label_memory[idx : idx + label_hat.size(0)] = label_hat.detach().cpu()
                    idx += normalized_sampled_features.size(0)
                    # memory.append(normalized_sampled_features.detach().cpu())

        if self.memory_size is None:
            self.feature_memory = torch.cat(feature_memory)
            self.label_memory = torch.cat(label_memory)
            if self.is_distributed:
                raise NotImplementedError("Distributed training is not supported for unbounded memory size")

        elif self.is_distributed:
            dist.barrier()
            self.feature_memory = self.feature_memory.to(self.device)
            self.label_memory = self.label_memory.to(self.device)
            receive_features = [torch.zeros_like(self.feature_memory) for _ in range(dist.get_world_size())]
            receive_labels = [torch.zeros_like(self.label_memory) for _ in range(dist.get_world_size())]
            dist.all_gather(receive_features, self.feature_memory)
            dist.all_gather(receive_labels, self.feature_memory)
            self.feature_memory = torch.cat(receive_features)
            self.label_memory = torch.cat(receive_labels)

    def save_memory(self) -> None:
        if self.is_distributed and dist.get_rank() != 0:
            return
        if self.f_mem_p is not None:
            torch.save(self.feature_memory.cpu(), self.f_mem_p)
        if self.l_mem_p is not None:
            torch.save(self.label_memory.cpu(), self.l_mem_p)

    def load_memory(self) -> bool:
        if self.f_mem_p is None:
            return False
        if os.path.isfile(self.f_mem_p) and os.path.isfile(self.l_mem_p):
            self.feature_memory = torch.load(self.f_mem_p).to(self.device)
            self.label_memory = torch.load(self.l_mem_p).to(self.device)
            return True
        return False

    def sample_features_batch(self, features: Tensor, pathified_gts: Tensor) -> Tuple[Tensor, Tensor]:
        the_max_value = self.num_classes
        if self.evaluation_task == "depth":
            # Here bin the continuous depth values into discrete bins
            # That are subsequently used to sample the features similarly
            # to the classes in the segmentation task
            pathified_gts = histogram_binning(pathified_gts, bins=self.num_bins)
            the_max_value = self.num_bins + 1

        batch_size = features.size(0)

        unique_classes_per_patch = batched_bincount(pathified_gts, max_value=the_max_value, dim=-1) > 0
        class_frequency = unique_classes_per_patch.sum(dim=(1, 2))
        patch_scores = (class_frequency.view(batch_size, 1, 1, -1) * unique_classes_per_patch).sum(-1).float()
        nonzero_indices = unique_classes_per_patch.sum(dim=-1) > 0

        patch_scores = patch_scores.flatten(start_dim=1)
        nonzero_indices = nonzero_indices.flatten(start_dim=1)
        patch_scores[~nonzero_indices] = 1e6

        uniform_x = torch.rand_like(patch_scores[nonzero_indices])
        patch_scores[nonzero_indices] *= uniform_x

        _, sampled_indices = torch.topk(patch_scores, self.num_sampled_features, largest=False)

        sampled_features = features.gather(1, sampled_indices.unsqueeze(-1).repeat(1, 1, features.shape[-1]))

        return sampled_features, sampled_indices

    def patchify_gt(self, gt: Tensor, patch_size: int) -> Tensor:
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h // patch_size, w // patch_size, c * patch_size * patch_size)
        return gt

    def patchify_depth(self, gt: Tensor, patch_size: int) -> Tensor:
        gt = gt[:, 0:1]  # we are storing 'gray scale' depth values
        bs, c, h, w = gt.shape
        gt = gt.reshape(bs, c, h // patch_size, patch_size, w // patch_size, patch_size)
        gt = gt.permute(0, 2, 4, 1, 3, 5)
        gt = gt.reshape(bs, h // patch_size, w // patch_size, c * patch_size * patch_size)
        return gt

    def cross_attention(self, q: Tensor, k: Tensor, v: Tensor, beta: float = 0.02) -> Tensor:
        """
        Args:
            q (torch.Tensor): query tensor of shape (bs, num_patches, d_k)
            k (torch.Tensor): key tensor of shape (bs, num_patches,  NN, d_k)
            v (torch.Tensor): value tensor of shape (bs, num_patches, NN, label_dim)
        """
        # d_k = q.size(-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        q = q.unsqueeze(2)  # (bs, num_patches, 1, d_k)
        attn = torch.einsum("bnld,bnmd->bnlm", q, k) / beta
        attn = attn.squeeze(2)
        attn = F.softmax(attn, dim=-1)
        attn = attn.unsqueeze(-1)
        label_hat = torch.einsum("blms,blmk->blsk", attn, v)
        label_hat = label_hat.squeeze(-2)
        return label_hat

    def find_nearest_key_to_query(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        bs, num_patches, d_k = q.shape
        reshaped_q = q.reshape(bs * num_patches, d_k)
        neighbors, distances = self.NN_algorithm.search_batched(reshaped_q)
        neighbors = neighbors.astype(np.int64)
        neighbors = torch.from_numpy(neighbors).to(self.device)
        neighbors = neighbors.flatten()
        key_features = self.feature_memory[neighbors]
        key_features = key_features.reshape(bs, num_patches, self.num_neighbour, -1)
        key_labels = self.label_memory[neighbors]
        key_labels = key_labels.reshape(bs, num_patches, self.num_neighbour, -1)
        return key_features, key_labels

    @torch.no_grad()
    def evaluate(
        self, val_loader: DataLoader, eval_spatial_resolution: Tuple[int, int], return_labels: bool = False
    ) -> Tuple[Dict[str, float | List[float]], Tensor]:
        if self.evaluation_task == "segmentation":
            # ...accumulated confusion matrices for class labels, valid pixels, and invalid pixels (in pixel counts)
            running_metric = torch.zeros((1, self.num_classes, self.num_classes), device=self.device)
        elif self.evaluation_task == "depth":
            running_metric = DepthMetrics(device=self.device)

        num_frames = 1
        label_hats = None
        start_idx = 0
        for _, batch in enumerate(tqdm(val_loader, desc="Evaluation loop")):
            x, y = batch["image"], batch[self.gt_key]

            if x.ndim == 5:
                # the model should handle the temporal dimension
                num_frames = y.shape[1]
                y = rearrange(y, "b t c h w -> (b t) c h w")

            x = x.to(self.device)
            *_, h, w = x.shape
            with torch.amp.autocast(self.device, dtype=self.dtype):
                features = self.ftr_extr_fn(x.to(self.device), inference=True)
            features = features.float()
            features = features.to(self.device)
            y = y.to(self.device)
            if self.evaluation_task == "segmentation":
                y = y.long()
            elif self.evaluation_task == "depth":
                y = y[:, 0:1]
            ## copy the data of features to another variable
            q = features.clone()
            q = q.detach().cpu().numpy()
            key_features, key_labels = self.find_nearest_key_to_query(q)
            label_hat = self.cross_attention(features, key_features, key_labels)
            bs, _, label_dim = label_hat.shape
            label_hat = label_hat.reshape(bs, eval_spatial_resolution[0], eval_spatial_resolution[1], label_dim).permute(
                0, 3, 1, 2
            )
            if self.evaluation_task == "depth":
                label_hat = label_hat.mean(dim=1, keepdim=True)
            resized_label_hats = F.interpolate(label_hat.float(), size=(h, w), mode="bilinear")
            if self.evaluation_task == "segmentation":
                cluster_map = resized_label_hats.argmax(dim=1).unsqueeze(1)

            if return_labels:
                if label_hats is None:
                    if self.evaluation_task == "segmentation":
                        size = (len(val_loader.dataset), num_frames, *cluster_map.shape[1:])
                    elif self.evaluation_task == "depth":
                        size = (len(val_loader.dataset), num_frames, *resized_label_hats.shape[1:])
                    label_hats = torch.empty(size, dtype=label_hat.dtype, device="cpu")

                if self.evaluation_task == "segmentation":
                    label_hats[start_idx : start_idx + cluster_map.size(0) // num_frames] = rearrange(
                        cluster_map.cpu(), "(b t) ... -> b t ...", t=num_frames
                    )
                    start_idx += cluster_map.size(0) // num_frames
                elif self.evaluation_task == "depth":
                    label_hats[start_idx : start_idx + resized_label_hats.size(0) // num_frames] = rearrange(
                        resized_label_hats.cpu(), "(b t) ... -> b t ...", t=num_frames
                    )
                    start_idx += resized_label_hats.size(0) // num_frames

            if self.evaluation_task == "segmentation":
                valid_idx = y != 255
                running_metric += fast_cm_torch(
                    y[valid_idx].unsqueeze(0), cluster_map[valid_idx].unsqueeze(0), self.num_classes, do_check=False
                )
            elif self.evaluation_task == "depth":
                running_metric.update(y, resized_label_hats)

        if self.is_distributed:
            dist.barrier()
            if self.evaluation_task == "segmentation":
                dist.all_reduce(running_metric, op=dist.ReduceOp.SUM)
            elif self.evaluation_task == "depth":
                running_metric.dist_all_reduce()

        if self.evaluation_task == "segmentation":
            # Metrics based on the ground-truth class labels <= cm, cm_valid, cm_invalid
            # ...mean intersection over union (mIoU) for all pixels, valid pixels, and invalid pixels
            iou_valid = per_class_iou_torch(running_metric).view(-1)
            miou_valid = torch.nanmean(iou_valid)
            logs = {"mIoU": miou_valid.item(), "IoU": iou_valid.tolist()}
        elif self.evaluation_task == "depth":
            logs = running_metric.compute()

        return logs, label_hats


def hbird_evaluation(
    *,
    ftr_extr_fn: FeaturesFunction,
    model_info: Info,
    train_dataset: Dataset,
    val_dataset: Dataset,
    dataset_info: Info,
    evaluation_task: str = "segmentation",
    num_bins: int = 255,
    batch_size: int = 64,
    batch_size_eval: int = 64,
    augmentation_epoch: int = 1,
    device: str = "cuda",
    dtype: str = "bf16",
    is_distributed: bool = False,
    num_workers: int = 8,
    return_labels: bool = False,
    num_neighbour: int = 30,
    nn_params: Kwargs = None,
    memory_size: Optional[int] = None,
    f_mem_p: Optional[str] = None,
    l_mem_p: Optional[str] = None,
) -> Tuple[Dict[str, float | List[float]], Tensor]:
    input_size = dataset_info["input_size"]
    patch_size = model_info["patch_size"]

    if isinstance(input_size, int):
        eval_spatial_resolution = (input_size // patch_size, input_size // patch_size)
    else:
        eval_spatial_resolution = (input_size[0] // patch_size, input_size[1] // patch_size)
    model_info["eval_spatial_resolution"] = eval_spatial_resolution

    dataset_size = dataset_info["dataset_size"]
    val_dataset_size = len(val_dataset)
    num_classes = dataset_info["num_classes"]
    window_size = dataset_info["window_size"]
    print("Train dataset size: ", dataset_size)
    print("Val dataset size: ", val_dataset_size)
    print("Number of classes: ", num_classes)
    print("Window size: ", window_size)

    if isinstance(memory_size, str):
        assert memory_size[0] == "x", "Memory size should be a string starting with x"
        memory_size = int(memory_size[1:]) * dataset_size * augmentation_epoch * window_size

    train_sampler = None if not is_distributed else DistributedSampler(train_dataset, shuffle=True)
    val_sampler = None if not is_distributed else DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not is_distributed,
        pin_memory=True,
        num_workers=num_workers,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_eval, shuffle=False, pin_memory=True, num_workers=num_workers, sampler=val_sampler
    )
    evaluator = HbirdEvaluation(
        ftr_extr_fn=ftr_extr_fn,
        model_info=model_info,
        train_loader=train_loader,
        dataset_info=dataset_info,
        num_neighbour=num_neighbour,
        augmentation_epoch=augmentation_epoch,
        evaluation_task=evaluation_task,
        num_bins=num_bins,
        device=device,
        dtype=dtype,
        is_distributed=is_distributed,
        nn_params=nn_params,
        memory_size=memory_size,
        f_mem_p=f_mem_p,
        l_mem_p=l_mem_p,
    )

    return evaluator.evaluate(val_loader, eval_spatial_resolution, return_labels=return_labels)


if __name__ == "__main__":
    import torch
    from torchvision.transforms import Normalize

    from vam.evaluation.datasets import CityscapesDataset, KITTIDataset

    DINOV = "v2"
    DTS = "cityscapes"
    scale = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # images are normalized to [-1, 1]

    target_size = (288, 512)
    if DINOV == "v1":
        vision_encoder = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        patch_size = 16

        def fwd(x: Tensor, inference: bool) -> Tensor:
            return vision_encoder.get_intermediate_layers(scale(x))[0][:, 1:]

    elif DINOV == "v2":
        vision_encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        patch_size = 14
        target_size = ((target_size[0] // 14) * 14, (target_size[1] // 14) * 14)

        def fwd(x: Tensor, inference: bool) -> Tensor:
            return vision_encoder(scale(x), is_training=True)["x_norm_patchtokens"]

    n_params = sum(p.numel() for p in vision_encoder.parameters())
    print("number of non-embedding parameters: %.2fM" % (n_params / 1e6,))
    vision_encoder = vision_encoder.to("cuda")
    vision_encoder = vision_encoder.eval()
    vision_encoder = vision_encoder.requires_grad_(False)
    model_info = {
        "patch_size": patch_size,
        "d_model": 768,
    }

    if DTS == "kitti":
        train_dts = KITTIDataset(root="/datasets_local/KITTI_STEP", split="train", target_size=target_size, window_size=1)
        val_dts = KITTIDataset(root="/datasets_local/KITTI_STEP", split="val", target_size=target_size, window_size=1)
    elif DTS == "cityscapes":
        train_dts = CityscapesDataset(root="/datasets_local/cityscapes", split="train", target_size=target_size)
        val_dts = CityscapesDataset(root="/datasets_local/cityscapes", split="val", target_size=target_size)

    dataset_info = {
        "dataset_size": len(train_dts),
        "num_classes": train_dts.get_num_classes(),
        "window_size": train_dts.get_window_size(),
        "input_size": train_dts.get_image_size(),
    }

    logs, preds = hbird_evaluation(
        ftr_extr_fn=fwd,
        model_info=model_info,
        train_dataset=train_dts,
        val_dataset=val_dts,
        dataset_info=dataset_info,
        batch_size=16,
        batch_size_eval=16,
        augmentation_epoch=1,
        device="cuda",
        return_labels=False,
        num_neighbour=30,
        nn_params=None,
        memory_size="x10",  # you can set this to reduce memory size
        f_mem_p=None,
        l_mem_p=None,
    )

    print(logs["IoU"], logs["mIoU"])
