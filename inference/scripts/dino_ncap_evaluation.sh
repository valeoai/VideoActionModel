SCRIPT_DIR=$(dirname $(realpath $0))

bash $SCRIPT_DIR/run_neuro_ncap_eval.sh \
$cya_ALL_CCFRSCRATCH/output_data/fbartocc/experiments/DINO_VAM/Dino_L_action_learning_8layers_nuPlan_nuScenes_0701_2231_1751401896/checkpoints/before_drop_epoch=000_step=0000006525.ckpt \
logs/dino_baseline_ncap

export NCAP_FOLDER=$cya_ALL_CCFRSCRATCH/ncap_eval/logs
OUTDIR=ncap_scores
mkdir -p $OUTDIR

# results[0]=$NCAP_FOLDER/dino_baseline_ncap/2025-01-22_13-55-09
# python $SCRIPT_DIR/aggregate_results.py --rootdir ${results[i]} > $OUTDIR/$(basename $(dirname ${results[i]})).log
# sbatch $SCRIPT_DIR/create_mp4.slurm ${results[i]}
