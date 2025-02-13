SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
TOKENIZER_JIT_PATH=$WORK/VQ_ds16_16384_llamagen.jit

BASE_DIR=$ycy_ALL_CCFRSCRATCH/nuscenes_v2
INPUT_DIR=$BASE_DIR/samples
FILES_LIST_DIR=$BASE_DIR/segments
OUTPUT_DIR=$BASE_DIR/tokens/samples
mkdir -p $FILES_LIST_DIR
mkdir -p $OUTPUT_DIR
INPUT_FILE=$FILES_LIST_DIR/frames_list.txt

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/video_action_model

mkdir -p hq_tokenize_nuscenes/tokens

rm -f $BASE_DIR/$INPUT_FILE

# Use find to get all frames
bash $SCRIPT_DIR/scripts/glob.sh $INPUT_DIR $INPUT_FILE


srun -A ycy@h100 -C h100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:1 --hint=nomultithread --qos=qos_gpu_h100-gc --time=00:30:00 \
python $SCRIPT_DIR/create_opendv_tokens.py \
--dataset nuscenes \
--queue sequential \
--frames_dir $FILES_LIST_DIR \
--outdir $OUTPUT_DIR \
--tokenizer_jit_path $TOKENIZER_JIT_PATH \
--num_cpus 16 \
--num_writer_threads 10 \
--writer_queue_size 10240 \
--batch_size 256 \
--dtype bf16

# Number of tokenized frames:
# find $ycy_ALL_CCFRSCRATCH/nuscenes_v2/tokens -type f -name "*.npy" | wc -l
