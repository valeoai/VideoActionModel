import itertools
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torchvision
from einops import rearrange
from PIL import Image
from torch import Tensor

Batch = Dict[str, Any]


class Detach:
    def __call__(self, image: Tensor) -> Tensor:
        return image.detach().cpu()


class ToChannelsLast:
    def __call__(self, image: Tensor) -> Tensor:
        return rearrange(image, "c ... -> ... c")


class NormalizeInverse:
    """
    Expect image input to be in [-1;1] rescale in [0;1]
    """

    def __call__(self, image: Tensor) -> Tensor:
        return (image + 1) / 2


denormalize_img = torchvision.transforms.Compose((Detach(), NormalizeInverse(), ToChannelsLast()))


def gridplot(
    img_list: List[Image.Image], titles: List[str] = [], cmaps: List[str] = [], cols: int = 2, figsize: Tuple[int] = (12, 12)
) -> plt.Figure:
    """
    Plot a list of images in a grid format

    Args:
        img_list: list of images to plot
        titles: list of titles to print above each image of `imgs`
        cols: number of column of the grid, the number of rows is determined accordingly
        figsize: matplotlib `figsize` figure param
    """

    rows = len(img_list) // cols + len(img_list) % cols

    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten()

    for img, title, cmap, ax in itertools.zip_longest(img_list, titles, cmaps, axs):
        if img is None:
            ax.set_visible(False)
            continue

        if img.ndim == 2 and cmap is None:
            cmap = "gray"

        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    return fig


def prepare_images_to_log(
    learning_phase: str, batch: Batch, preds: Tensor, batch_idx: int, log_images_interval: int
) -> Dict[str, plt.Figure]:
    if log_images_interval == 0 or batch_idx % log_images_interval != 0:
        return {}

    i = 0  # for each batch, we always log the first image of the batch only

    prefix = f"{learning_phase}/batch_{batch_idx}"

    img_list = [denormalize_img(preds[i]), denormalize_img(batch["images"][i])]
    titles = ["estimation", "GT"]
    cmaps = [None, None]

    colums = 2
    plt_figure = gridplot(
        img_list,
        titles=titles,
        cmaps=cmaps,
        cols=colums,
        figsize=(6 * colums, 6 * ((len(img_list) + 1) // colums)),
    )

    return {prefix: plt_figure}
