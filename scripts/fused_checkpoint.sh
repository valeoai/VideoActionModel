module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/worldmodel
export TRITON_CACHE_DIR=$SCRATCH/.triton

SCRIPT_DIR=$(dirname "$(realpath "$0")")

INPUT=$1
OUTPUT=$2


srun --qos=qos_gpu_h100-dev -A ycy@h100 -C h100 --time=00:30:00 --gres=gpu:1  --ntasks=1 --cpus-per-task=48 --pty \
python $SCRIPT_DIR/fused_checkpoint.py --checkpoint $INPUT --output $OUTPUT

# Example usage:
# bash scripts/fused_checkpoint.sh \
# $ycy_ALL_CCFRSCRATCH/output_data/opendv_gpt2_LlamaGen/grid_search/GS256024_Nodes6_BSperGPU16_totalBS384_dim256_std0.0056_lr0.0003_0115_1029_1736933366/csv/version_0/checkpoints/'epoch=0-step=15529.ckpt' \
# $ycy_ALL_CCFRSCRATCH/test_fused_checkpoint/'epoch=0-step=15529_fused.ckpt'
