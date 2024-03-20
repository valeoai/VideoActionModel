from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from world_model.utils.cmd_line_logging import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    config: DictConfig,
    print_order: Sequence[str] = [],
    resolve: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        config: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are printed. Default is ``("data", "model",
            "callbacks", "logger", "trainer", "paths")``.
        resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in config else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in config:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)