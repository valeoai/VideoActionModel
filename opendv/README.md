# Preparing OpenDV dataset

## Downloading OpenDV dataset

```bash
wget
```

## Extracting frames

```bash
mkdir -p './!{video_id}'

ffmpeg -ss '!{discard_start}' \
        -i './!{input_video}' \
        -vf "fps=!{params.fps},scale=!{params.frame_w}:!{params.frame_h}" \
        -q:v 2 \
        './!{video_id}/f_%06d.jpg' &> frames.log

tail_end=$(cat frames.log | grep -Eo 'frame= *[0-9]+ *' | grep -Eo '[0-9]+' | tail -1)
if [ -z "$tail_end" ]; then
    echo "Error: Could not extract frames from !{input_video}" >&2
    exit 1
fi
tail_start=$((tail_end - !{discard_end}*!{params.fps}))
echo -e "tail_start: $tail_start\ntail_end: $tail_end\nlast_frame:\n$((tail_start-1))" > './!{video_id}/frame_count'

for i in $(seq -f "%06g" $tail_start $tail_end); do
    rm "./!{video_id}/f_$i.jpg"
done
```

## Preparint the tokens

```bash
```
