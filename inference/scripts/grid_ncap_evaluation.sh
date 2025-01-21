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
echo $SCRIPT_DIR

for i in {0..4}; do
    echo "Running evaluation for ${ckpt[i]}"
    bash $SCRIPT_DIR/run_neuro_ncap_eval.sh ${ckpt[i]} ${log_dir[i]}
done
