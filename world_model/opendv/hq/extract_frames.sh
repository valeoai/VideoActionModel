#!/bin/bash
CSV_FILE=$1
VIDEO_PATH=$2
FPS=$3
WIDTH=$4
HEIGHT=$5
OUTDIR=$6

# Get video id
VIDEO_ID=$(basename "${VIDEO_PATH%%.*}")
# replace leading "-" with "@" as this is done in the metadata file
VIDEO_ID="${VIDEO_ID/#-/@}"
echo "VIDEO_ID: $VIDEO_ID"
# Directory
VIDEO_DIR=$(basename $(dirname "${VIDEO_PATH}"))
OUTDIR=${OUTDIR}/${VIDEO_DIR}/${VIDEO_ID}
mkdir -p ${OUTDIR}

# Initialize variables with default values
DISCARD_START=0
DISCARD_END=0

# Using awk to extract values and storing them in variables
eval $(awk '
    BEGIN { FS="\t" }  # Set field separator to tab
    NR==1 {
        for (i=1; i<=NF; i++) {
            gsub(/^[ \t]+|[ \t]+$/, "", $i)
            if ($i == "video_id") {
                vid_col = i
            }
            if ($i == "discard_start") {
                start_col = i
            }
            if ($i == "discard_end") {
                end_col = i
            }
        }
    }
    NR>1 {
        gsub(/^[ \t]+|[ \t]+$/, "", $vid_col)
        if ($vid_col == "'$VIDEO_ID'") {
            gsub(/^[ \t]+|[ \t]+$/, "", $start_col)
            gsub(/^[ \t]+|[ \t]+$/, "", $end_col)
            print "DISCARD_START=" $start_col
            print "DISCARD_END=" $end_col
            exit
        }
    }
' "$CSV_FILE")

# Now you can use the variables
echo "Discard Start: $DISCARD_START"
echo "Discard End: $DISCARD_END"

# Get video duration
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${VIDEO_PATH}")
# Remove discarded seconds from the end
END_TIME=$(echo $duration - $DISCARD_END | bc)

echo "Duration: $duration"
echo "End time: $END_TIME"

# Extract frames from video
ffmpeg -hide_banner -loglevel error  \
-ss "${DISCARD_START}" -i "${VIDEO_PATH}" -to "${END_TIME}" \
-vf "fps=${FPS},scale=${WIDTH}:${HEIGHT}" \
-q:v 2 -vsync 0 -frame_pts 0 -movflags +faststart \
-y "${OUTDIR}/f_%06d.jpg"
