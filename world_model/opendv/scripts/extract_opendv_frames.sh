SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
NUM_WORKERS=1
CPUS_PER_WORKER=24

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


# find all mp4 and webm files in the input directory
# find $INPUT_DIR -type f -name "*.mp4" -o -name "*.webm" | sort > $BASE_DIR/videos.txt

# extract frames from each video
# while read -r VIDEO_PATH; do
#     echo "Extracting frames from $VIDEO_PATH"
#     hq submit --cpus $CPUS_PER_WORKER \
#     bash $SCRIPT_DIR/scripts/_extract_frames.sh \
#     $CSV_FILE \
#     $VIDEO_PATH \
#     $FPS \
#     $WIDTH \
#     $HEIGHT \
#     $OUTDIR
# done < $BASE_DIR/videos.txt


# find $fzh_ALL_CCFRSCRATCH/OpenDV_release/frames512 -type f -name "*.jpg" | wc -l
# find  $fzh_ALL_CCFRSCRATCH/OpenDV_release/frames512 -maxdepth 2 -type d  | wc -l
# hq server stop
# find . -name "0.stdout" -exec grep -H "6rwRHG_PG" {} \;
# find $fzh_ALL_CCFRSCRATCH/OpenDV_Youtube/videos -type f -name "*.mp4" -o -name "*.webm" | wc -l

# find $fzh_ALL_CCFRSCRATCH/OpenDV_release/frames512 -type f -name "*.jpg" | wc -l
# find job* -name "0.stderr" -type f -not -empty
