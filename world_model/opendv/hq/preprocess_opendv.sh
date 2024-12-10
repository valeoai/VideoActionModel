#!/bin/bash
hq server start --idle-timeout=300sec &

hq server info  # CAUTION: Ensure that the server is running before launching the workers!

for gpu in {0..0}; do
    CUDA_VISIBLE_DEVICES=$gpu hq worker start --no-detect-resources --idle-timeout 300sec --on-server-lost finish-running --cpus 4 --resource "gpus/nvidia=sum(1)" &
done


python world_model/opendv/create_opendv_tokens.py \
--video_list world_model/opendv/opendv_video.json \
--metadata ~/iveco/datasets_iveco_raw/OpenDV_Youtube/videos_metadata.csv \
--outdir ~/data/OpenDV_Youtube/tokens \
--tmpdir ~/data/OpenDV_Youtube/tmp \
--tokenizer_jit_path ~/iveco/scratch_iveco/world_model_JZGC4/jit_models/VQ_ds16_16384_llamagen.jit \
--num_writer_threads 5 \
--frames_queue_size 10000 \
--writer_queue_size 10000 \
--batch_size 64 \
--target_frame_rate 5 \
--target_width 512 \
--target_height 288
