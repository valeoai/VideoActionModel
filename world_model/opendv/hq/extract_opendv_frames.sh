SCRIPT_DIR=$(dirname $(dirname "$(realpath "$0")"))
NUM_WORKERS=1
CPUS_PER_WORKER=2

FPS=5
WIDTH=512
HEIGHT=288

INPUT_DIR=$ycy_ALL_CCFRSTORE/OpenDV_Youtube/videos
CSV_FILE=$ycy_ALL_CCFRSTORE/OpenDV_Youtube_meta/videos_metadata.csv
BASE_DIR=$fzh_ALL_CCFRSCRATCH/OpenDV_release/
OUTPUT_DIR=$BASE_DIR/frames512
mkdir -p $OUTPUT_DIR

bash $SCRIPT_DIR/hq/start_hq_archive_slurm.sh $NUM_WORKERS $CPUS_PER_WORKER

# find all mp4 and webm files in the input directory
find $INPUT_DIR -type f -name "*.mp4" -o -name "*.webm" > $BASE_DIR/videos.txt

# extract frames from each video
while read -r VIDEO_PATH; do
    echo "Extracting frames from $VIDEO_PATH"
    hq submit --cpus $CPUS_PER_WORKER \
    bash $SCRIPT_DIR/hq/extract_frames.sh \
    $CSV_FILE \
    $VIDEO_PATH \
    $FPS \
    $WIDTH \
    $HEIGHT \
    $OUTPUT_DIR
done < $BASE_DIR/videos.txt


# hq server stop --server-dir ~/.hq_server
