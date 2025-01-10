# Number of CPUs per worker
if [ -z "$1" ]; then
    echo "Usage: $0 <CPUS_PER_WORKER>"
    exit 1
fi
CPUS_PER_WORKER=$1

SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
INPUT_FILE=frames_list.txt
TOKENIZER_JIT_PATH=$WORK/VQ_ds16_16384_llamagen.jit

INPUT_DIR=$fzh_ALL_CCFRSCRATCH/nuscenes_v2/samples
BASE_DIR=$ycy_ALL_CCFRSCRATCH/nuscenes_v2
FILES_LIST_DIR=$BASE_DIR/segments
OUTPUT_DIR=$BASE_DIR/tokens_samples
mkdir -p $BASE_DIR
chmod g+rwxs,o+rx $BASE_DIR
setfacl -d -m g::rwx $BASE_DIR
mkdir -p $FILES_LIST_DIR
mkdir -p $OUTPUT_DIR

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

mkdir -p .hq-server
# Set the directory which hyperqueue will use
export HQ_SERVER_DIR=${PWD}/.hq-server

bash $SCRIPT_DIR/hq/start_hq_slurm.sh 1 $CPUS_PER_WORKER

mkdir -p hq_tokenize_nuscenes/tokens

rm -f $BASE_DIR/$INPUT_FILE

# Use find to get all frames
hq submit --name=GLOB --wait --stdout 'hq_tokenize_nuscenes/glob_%{JOB_ID}.stdout' --stderr 'hq_tokenize_nuscenes/glob_%{JOB_ID}.stdout' \
bash $SCRIPT_DIR/scripts/glob.sh $INPUT_DIR $FILES_LIST_DIR/$INPUT_FILE


srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=5 --gres=gpu:0 --hint=nomultithread --qos=qos_gpu_h100-gc --time=10:00:00 \
python $SCRIPT_DIR/create_opendv_tokens.py \
--dataset nuscenes \
--server_dir $HQ_SERVER_DIR \
--frames_dir $FILES_LIST_DIR \
--outdir $OUTPUT_DIR \
--tokenizer_jit_path $TOKENIZER_JIT_PATH \
--num_cpus $CPUS_PER_WORKER \
--num_writer_threads 10 \
--writer_queue_size 10240 \
--batch_size 256 \
--dtype bf16

hq server stop

# Find failed jobs:
# find hq_tokenize_opendv/tokens -name "*.err" -type f -not -empty

# Number of created directories:
# find  $fzh_ALL_CCFRSCRATCH/OpenDV_processed/tokens -mindepth 2 -maxdepth 2 -type d  | wc -l

# Number of tokenized frames:
# find $fzh_ALL_CCFRSCRATCH/OpenDV_processed/tokens -type f -name "*.npy" | wc -l
