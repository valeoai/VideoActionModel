SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
NUM_WORKERS=16
CPUS_PER_WORKER=10
MAX_NUM_FILES=10000
INPUT_FILE=frames_list.txt

INPUT_DIR=$ycy_ALL_CCFRSCRATCH/OpenDV/frames512
BASE_DIR=$ycy_ALL_CCFRSCRATCH/OpenDV_release
FILES_LIST_DIR=$BASE_DIR/segments
OUTPUT_DIR=$BASE_DIR/tokens
TOKENIZER_JIT_PATH=$WORK/VQ_ds16_16384_llamagen.jit
mkdir -p $FILES_LIST_DIR
mkdir -p $OUTPUT_DIR

module purge
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

bash $SCRIPT_DIR/hq/start_hq_slurm.sh $NUM_WORKERS $CPUS_PER_WORKER

# Use find to find all frames
# hq submit --wait \
# bash $SCRIPT_DIR/hq/glob.sh $INPUT_DIR $BASE_DIR/$INPUT_FILE

# # Split the file
# hq submit --wait \
# split -l "$MAX_NUM_FILES" "$BASE_DIR/$INPUT_FILE" "$FILES_LIST_DIR/segment_"

# # Rename files to have .txt extension and proper numbering
# count=1
# for file in "$FILES_LIST_DIR"/segment_*; do
#     new_name="$FILES_LIST_DIR/segment_$(printf "%06d" $count).txt"
#     mv "$file" "$new_name"
#     count=$((count + 1))
# done

# num_segments=$(ls -1q $FILES_LIST_DIR | wc -l)

# echo "Split $INPUT_FILE into $num_segments segments in $FILES_LIST_DIR/"
# echo "Each file contains up to $MAX_NUM_FILES lines"


srun -A fzh@v100 -C v100 --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:0 --hint=nomultithread --qos=qos_gpu-t3 --time=10:00:00 \
python $SCRIPT_DIR/create_opendv_tokens.py \
--server_dir $WORK/.hq_server \
--frames_dir $FILES_LIST_DIR \
--outdir $OUTPUT_DIR \
--tokenizer_jit_path $TOKENIZER_JIT_PATH \
--num_cpus $CPUS_PER_WORKER \
--num_writer_threads 5 \
--frames_queue_size 10000 \
--writer_queue_size 10000 \
--batch_size 16

hq server stop --server-dir ~/.hq_server
