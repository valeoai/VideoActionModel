# Evaluation cripts

This folder contains most of the scripts we used to run our evaluations. The SLURM files are taylored to the Jean-Zay cluster, but should work after some modifications on other clusters.

Here is a non-exhaustive list of the scripts available:

```bash
scripts
|––––evaluate_ego_trajectory.[py,slurm]  # Evaluate minADE (open-loop metric)
|––––hbird_evaluation.[py,slurm]  # Use humming-bird to evaluate the model in semantic segmentation or depth estimation
|––––nxt_evaluation.[py,slurm]  # Evaluate GPT models for next token prediction
|––––quality_evaluation.[py,slurm]  # Evaluate the quality of the generated videos with FID
|––––video_qualitative_results.py  # Generate videos from the GPT model
```

## Config

In the `VideoActionModel/configs/paths` folder you can find the configuration files to the different paths required to run all experiments, e.g. `VideoActionModel/configs/paths/eval_paths_jeanzay.yaml`. To run experiment in your environment, you can add a file with your paths and use it as an argument in the different scripts, by adding it to the command line:

```bash
--config configs/paths/eval_paths_custom.yaml
```

## Running the scripts

The scripts were tested when ran from the root of the repository, i.e.:

```bash
cd VideoActionModel
[python,sbatch] scripts/...[py,slurm] ...
```
