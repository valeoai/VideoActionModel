# Set the number of workers and cpus per worker
# Number of workers
if [ -z "$1" ]; then
    echo "Usage: $0 <NUM_WORKERS> <CPUS_PER_WORKER>"
    exit 1
fi
NUM_WORKERS=$1

# Number of CPUs per worker
if [ -z "$2" ]; then
    echo "Usage: $0 <NUM_WORKERS> <CPUS_PER_WORKER>"
    exit 1
fi
CPUS_PER_WORKER=$2

SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
FPS=10
WIDTH=512
HEIGHT=288

INPUT_DIR=$fzh_ALL_CCFRSCRATCH/OpenDV_Youtube/videos
CSV_FILE=$fzh_ALL_CCFRSCRATCH/OpenDV_Youtube_meta/videos_metadata.csv
BASE_DIR=$fzh_ALL_CCFRSCRATCH/OpenDV_release
OUTDIR=$BASE_DIR/frames512
mkdir -p $OUTDIR
chmod g+rwxs,o+rx $OUTDIR
setfacl -d -m g::rwx $OUTDIR

mkdir -p .hq-server
# Set the directory which hyperqueue will use
export HQ_SERVER_DIR=${PWD}/.hq-server

# Start the hyperqueue server & workers without GPUs
bash $SCRIPT_DIR/hq/start_hq_archive_slurm.sh $NUM_WORKERS $CPUS_PER_WORKER

OUTPUT_FILE="${PWD}/tasks.toml"
rm -f "$OUTPUT_FILE"

echo 'name = "extract_opendv_frames"' > "$OUTPUT_FILE"
echo 'stream_log = "extract_opendv_frames.log"' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

find "$INPUT_DIR" -type f -name "*.mp4" -o -name "*.webm" | sort | while read -r file; do
    cat >> "$OUTPUT_FILE" << EOF
[[task]]
stdout = "task_%{TASK_ID}.out"
stderr = "task_%{TASK_ID}.err"
command = ["bash", "${SCRIPT_DIR}/scripts/_extract_frames.sh", "${CSV_FILE}", "${file}", "${FPS}", "${WIDTH}", "${HEIGHT}", "${OUTDIR}"]

[[task.request]]
resources = { "cpus" = "${CPUS_PER_WORKER}" }

EOF
done

hq job submit-file "$OUTPUT_FILE"
# hq server stop

# Some helpful commands to check the extractions:

# Number of videos:
# find $fzh_ALL_CCFRSCRATCH/OpenDV_Youtube/videos -type f -name "*.mp4" -o -name "*.webm" | wc -l

# Number of extracted frames:
# find $fzh_ALL_CCFRSCRATCH/OpenDV_release/frames512 -type f -name "*.jpg" | wc -l

# Number of created directories:
# find  $fzh_ALL_CCFRSCRATCH/OpenDV_release/frames512 -maxdepth 2 -type d  | wc -l

# Find the log file of a specific job:
# find . -name "0.stdout" -exec grep -H "6rwRHG_PG" {} \;

# Find failed jobs:
# find job* -name "0.stderr" -type f -not -empty
