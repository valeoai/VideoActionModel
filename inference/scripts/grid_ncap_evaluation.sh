module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0
export PYTHONUSERBASE=$WORK/python_envs/world_model

ckpt[0]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000038823_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0121_0053_1737417196/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[0]=logs/action_expert_38k_attndim768_actdim192

ckpt[1]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000077646_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0121_0112_1737418368/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[1]=logs/action_expert_77k_attndim768_actdim192

ckpt[2]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000116469_DDP_Nodes6_BSperGPU16_totalBS384_attdim768_actdim192_0121_0112_1737418377/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[2]=logs/action_expert_116k_attndim768_actdim192

ckpt[3]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000038823_DDP_Nodes6_BSperGPU16_totalBS384_attdim1024_actdim256_0121_0112_1737418373/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[3]=logs/action_expert_38k_attndim1024_actdim256

ckpt[4]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000139763_DDP_Nodes12_BSperGPU8_totalBS384_attdim2048_actdim512_0121_2057_1737489441/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[4]=logs/action_expert_139k_attndim2048_actdim512

ckpt[5]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000077646_DDP_Nodes6_BSperGPU16_totalBS384_attdim1024_actdim256_0121_0052_1737417153/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[5]=logs/action_expert_77k_attndim1024_actdim256


SCRIPT_DIR=$(dirname $(realpath $0))

# for i in {1..4}; do
#     echo "Running evaluation for ${ckpt[i]}"
#     bash $SCRIPT_DIR/run_neuro_ncap_eval.sh ${ckpt[i]} ${log_dir[i]}
# done

export NCAP_FOLDER=$ycy_ALL_CCFRSCRATCH/ncap_eval/logs
OUTDIR=ncap_scores
mkdir -p $OUTDIR
results[0]=$NCAP_FOLDER/action_expert_38k_attndim768_actdim192/2025-01-22_13-55-09
results[1]=$NCAP_FOLDER/action_expert_77k_attndim768_actdim192/2025-01-22_14-42-31
results[2]=$NCAP_FOLDER/action_expert_116k_attndim768_actdim192/2025-01-22_14-42-32
results[3]=$NCAP_FOLDER/action_expert_38k_attndim1024_actdim256/2025-01-22_14-42-34
results[4]=$NCAP_FOLDER/action_expert_139k_attndim2048_actdim512/2025-01-22_14-42-37
results[5]=$NCAP_FOLDER/action_expert_77k_attndim1024_actdim256/


for i in {0..4}; do
    echo "Aggregating evaluation for ${ckpt[i]}"
    python $SCRIPT_DIR/aggregate_results.py --rootdir ${results[i]} > $OUTDIR/$(basename $(dirname ${results[i]})).log
    # sbatch $SCRIPT_DIR/create_mp4.slurm ${results[i]}
done


# python inference/scripts/aggregate_results.py --rootdir $WORK/neuro-ncap/logs/action_expert_77k_attndim768_actdim192/2025-01-22_10-27-46
# python inference/scripts/create_mp4.py --rootdir $ycy_ALL_CCFRSCRATCH/ncap_eval/logs/action_expert_139k_attndim2048_actdim512/2025-01-22_14-42-37
# sbatch inference/scripts/create_mp4.slurm $ycy_ALL_CCFRSCRATCH/ncap_eval/logs/action_expert_139k_attndim2048_actdim512/2025-01-22_14-42-37
