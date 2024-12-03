# Preparing OpenDV dataset

## Download the dataset

First follow the instructions in the [OpenDV Dataset](https://github.com/OpenDriveLab/DriveAGI) repository to download the data.

## Donwload the metadata

```bash
wget https://www.rocq.inria.fr/cluster-willow/amiech/OpenDV_Youtube_metadata.json
```

## Prepare the dataset

Then you need to create the following folder structure:

```bash
OpenDV_Youtube
|––––videos
|----------Driver1
|-----------------video1.mp4
|-----------------video2.webm
|-----------------...
|----------Driver2
|----------...
|––––metadata.csv
```

## Tokenizer

Then download the llamagen tokenizer jit file:

```bash
wget https://www.rocq.inria.fr/cluster-willow/amiech/llamagen.jit
```

## Preparint the tokens

If you have jq installed, you can use the following command to create the video list

```bash
find ~/iveco/datasets_iveco_raw/OpenDV_Youtube/videos \
-type f \( -name "*.mp4" -o -name "*.webm" \) -print0 | \
jq -R -s 'split("\u0000")[:-1]' | \
jq . > opendv/opendv_video.json
```

Then you can use the following command to create the tokens:

```bash
python world_model/opendv/create_opendv_tokens.py \
--video_list world_model/opendv/opendv_video.json \
--metadata ~/iveco/datasets_iveco_raw/OpenDV_Youtube/videos_metadata.csv \
--outdir ~/data/OpenDV_Youtube/tokens \
--tmpdir ~/data/OpenDV_Youtube/tmp \
--tokenizer_jit_path ~/iveco/scratch_iveco/world_model_JZGC4/jit_models/VQ_ds16_16384_llamagen.jit \
--num_frames_threads 1 \
--num_writer_threads 1 \
--frames_queue_size 10000 \
--writer_queue_size 10000 \
--batch_size 64 \
--target_frame_rate 5 \
--target_width 512 \
--target_height 288
```
