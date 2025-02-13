from typing import Any, Dict

import yaml

from vam.utils.expand_path import expand_path


def read_eval_config(config_path: str) -> Dict[str, Any]:
    with open(expand_path(config_path), "r") as f:
        config = yaml.safe_load(f)
    return config
