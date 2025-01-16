module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

export HYDRA_FULL_ERROR=1

srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 --hint=nomultithread --qos=qos_gpu_h100-dev --time=00:10:00 \
python world_model/train.py \
 experiment=action_learning  \
 paths.output_dir=$SCRATCH/WM_debug/ \
 data.batch_size=12 \
 data.num_workers=12 \
 model.vai0rbis_conf.gpt_checkpoint_path=$ycy_ALL_CCFRSCRATCH/test_fused_checkpoint/epoch_0-step_15529_fused.ckpt \
 trainer=gpu \
 trainer.precision="bf16-mixed" \
 ++trainer.max_epochs=1 \
 ++trainer.limit_train_batches=100 \
 ++trainer.limit_val_batches=2


# srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 --hint=nomultithread --qos=qos_gpu_h100-dev --time=00:45:00 bash
