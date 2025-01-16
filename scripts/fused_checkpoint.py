import argparse
import os

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


def _path(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


parser = argparse.ArgumentParser(description="Fuse checkpoint files")
parser.add_argument("--checkpoint", type=_path, required=True, help="Path to the checkpoint file")
parser.add_argument("--output", type=_path, required=True, help="Path to the output file")
args = parser.parse_args()

convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint, args.output)
