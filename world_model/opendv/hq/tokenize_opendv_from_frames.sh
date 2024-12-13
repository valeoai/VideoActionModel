SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
NUM_WORKERS=1
CPUS_PER_WORKER=10

INPUT_DIR=$ycy_ALL_CCFRSCRATCH/OpenDV/frames512
OUTPUT_DIR=$fzh_ALL_CCFRSCRATCH/OpenDV/tokens
TOKENIZER_JIT_PATH=$WORK/VQ_ds16_16384_llamagen.jit
mkdir -p $OUTPUT_DIR

module purge
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

# bash $SCRIPT_DIR/hq/start_hq_slurm.sh $NUM_WORKERS $CPUS_PER_WORKER

srun -A fzh@v100 -C v100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --qos=qos_gpu-dev --time=00:10:00 \
python $SCRIPT_DIR/create_opendv_tokens.py \
--server_dir $WORK/.hq_server \
--frames_dir $INPUT_DIR \
--size_of_chunks 1000000 \
--outdir $OUTPUT_DIR \
--tokenizer_jit_path $TOKENIZER_JIT_PATH \
--num_writer_threads 5 \
--frames_queue_size 10000 \
--writer_queue_size 10000 \
--batch_size 64

hq server stop --server-dir ~/.hq_server
