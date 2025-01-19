module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model
export TRITON_CACHE_DIR=$SCRATCH/.triton

SCRIPT_DIR=$(dirname "$(realpath "$0")")

INPUT=$1
OUTPUT=$2


srun --qos=qos_gpu_h100-dev -A ycy@h100 -C h100 --time=00:30:00 --gres=gpu:1  --ntasks=1 --cpus-per-task=24 --pty \
python $SCRIPT_DIR/fused_checkpoint.py --checkpoint $INPUT --output $OUTPUT

# Example usage:
# bash scripts/fused_checkpoint.sh \
# $ycy_ALL_CCFRSCRATCH/output_data/vaiorbis_grid_search/Vaiorbis_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0119_1044_1737279899/checkpoints/'before_drop_epoch=000_step=0000006525.ckpt' \
# $ycy_ALL_CCFRSCRATCH/test_fused_checkpoint/tmp_action_expert_fused.pt


python scripts/fused_checkpoint.py \
--checkpoint $ycy_ALL_CCFRSCRATCH/output_data/vaiorbis_grid_search/Vaiorbis_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0119_1044_1737279899/checkpoints/'before_drop_epoch=000_step=0000006525.ckpt' \
--output $ycy_ALL_CCFRSCRATCH/test_fused_checkpoint/tmp_action_expert_fused.pt
