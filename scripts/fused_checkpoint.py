"""
Example usage:
srun -A cya@h100 -C h100 --pty \
--nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread \
--qos=qos_gpu_h100-dev --time=00:15:00 \
python scripts/fused_checkpoint.py \
--checkpoint $SCRATCH/2048_pretrain \
--output $SCRATCH/2048_pretrain_fused.pt
"""
import argparse

from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

from vam.utils import expand_path


parser = argparse.ArgumentParser(description="Fuse checkpoint files")
parser.add_argument("--checkpoint", type=expand_path, required=True, help="Path to the checkpoint file")
parser.add_argument("--output", type=expand_path, required=True, help="Path to the output file")
args = parser.parse_args()

convert_zero_checkpoint_to_fp32_state_dict(args.checkpoint, args.output)
