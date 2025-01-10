# Set the number of workers and cpus per worker
# Number of workers
if [ -z "$1" ]; then
    echo "Usage: $0 <NUM_WORKERS>"
    exit 1
fi
NUM_WORKERS=$1

SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))

INPUT_DIR="${ycy_ALL_CCFRSTORE}/nuplan/archives/sensor_blobs"
OUTDIR=$fzh_ALL_CCFRSCRATCH/nuplan_v2
mkdir -p $OUTDIR
chmod g+rwxs,o+rx $OUTDIR
setfacl -d -m g::rwx $OUTDIR

mkdir -p .hq-server
# Set the directory which hyperqueue will use
export HQ_SERVER_DIR=${PWD}/.hq-server

mkdir -p extract_nuplan_archive

# Start the hyperqueue server & workers without GPUs
# bash $SCRIPT_DIR/hq/start_hq_prepost_slurm.sh $NUM_WORKERS

OUTPUT_FILE="tasks.toml"
rm -f "$OUTPUT_FILE"

echo 'name = "extract_nuplan_archive"' > "$OUTPUT_FILE"
echo 'stream_log = "extract_nuplan_archive.log"' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

find "${INPUT_DIR}" -type f -name "*.zip" | sort | while read -r file; do
    cat >> "$OUTPUT_FILE" << EOF
[[task]]
stdout = "extract_nuplan_archive/task_%{TASK_ID}.out"
stderr = "extract_nuplan_archive/task_%{TASK_ID}.err"
command = ["tar", "-xvf", "${file}", "-C", "${OUTDIR}"]

[[task.request]]
resources = { "cpus" = "1" }

EOF
done

hq job submit-file "$OUTPUT_FILE"
