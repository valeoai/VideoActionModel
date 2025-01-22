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

ckpt[4]=$ycy_ALL_CCFRSCRATCH/output_data/vaiorbis/Vaiorbis_pretrained0000077646_DDP_Nodes6_BSperGPU16_totalBS384_attdim1024_actdim256_0121_0052_1737417153/checkpoints/end_of_epoch_epoch=000_step=0000007251.ckpt
log_dir[4]=logs/action_expert_77k_attndim1024_actdim256

SCRIPT_DIR=$(dirname $(realpath $0))

for i in {0..3}; do
    echo "Running evaluation for ${ckpt[i]}"
    bash $SCRIPT_DIR/run_neuro_ncap_eval.sh ${ckpt[i]} ${log_dir[i]}
done

NCAP_FOLDER=$WORK/neuro-ncap
OUTDIR=ncap_scores
mkdir -p $OUTDIR
results[0]=$NCAP_FOLDER/${log_dir[0]}/
results[1]=$NCAP_FOLDER/${log_dir[1]}/
results[2]=$NCAP_FOLDER/${log_dir[2]}/
results[3]=$NCAP_FOLDER/${log_dir[3]}/
results[4]=$NCAP_FOLDER/${log_dir[4]}/2025-01-21_20-54-08


# for i in {4..4}; do
#     echo "Aggregating evaluation for ${ckpt[i]}"
#     python $SCRIPT_DIR/aggregate_results.py --rootdir ${results[i]} > $OUTDIR/$(basename $(dirname ${results[i]})).log
# done
