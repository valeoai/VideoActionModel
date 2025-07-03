SCRIPT_DIR=$(dirname $(realpath $0))

bash $SCRIPT_DIR/run_neuro_ncap_eval.sh \
$cya_ALL_CCFRSCRATCH/output_data/fbartocc/experiments/DINO_VAM/Dino_L_action_learning_8layers_nuPlan_nuScenes_0701_2231_1751401892/checkpoints/end_of_epoch_epoch=000_step=0000007329.ckpt \
logs/dino_baseline_ncap

export NCAP_FOLDER=$cya_ALL_CCFRSCRATCH/ncap_eval/logs
export OUTDIR=ncap_scores
mkdir -p $OUTDIR

# export results=$NCAP_FOLDER/dino_baseline_ncap/2025-07-03_08-58-47
# python inference/scripts/aggregate_results.py --rootdir ${results} > dino_baseline_ncap_scores.log
# sbatch inference/scripts/create_mp4.slurm ${results}
