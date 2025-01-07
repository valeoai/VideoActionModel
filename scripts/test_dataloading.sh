module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

SCRIPT_DIR=$(dirname "$(realpath "$0")")

srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --qos=qos_gpu_h100-dev --time=00:20:00 \
python $SCRIPT_DIR/test_dataloading.py


# srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --qos=qos_gpu_h100-dev --time=00:45:00 bash
