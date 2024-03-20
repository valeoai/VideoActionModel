import os
from argparse import ArgumentParser
from pathlib import Path

WORK_DIR = Path(os.path.expandvars("$SHAREDWORKDIR"))
REPO_DIR = WORK_DIR / "code/NextTokenPrediction/"
ENVS_ROOT_DIR = WORK_DIR / "code/env"

if __name__ == "__main__":
    parser = ArgumentParser(prog="PROG")
    parser.add_argument("--run_name", "-n", type=str, required=True)
    parser.add_argument("--python_cmd", "-p", type=str, required=True)
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--cpus_per_node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--wall_time", "-wt", type=int, default=20)
    parser.add_argument("--allow_hyper_threading", action='store_true')
    # parse params
    args = parser.parse_args()

    print("RUN NAME: ", args.run_name)

    devices_args = f'++trainer.accelerator="gpu" ++trainer.devices={args.gpus_per_node} ++trainer.num_nodes={args.nodes}'
    
    if args.nodes * args.gpus_per_node > 1:
        devices_args = "trainer=ddp " +  devices_args
    
    slurm_ressources = f"--ntasks={args.nodes * args.gpus_per_node} --ntasks-per-node={args.gpus_per_node} --cpus-per-task={args.cpus_per_node}"
    if not args.allow_hyper_threading:
        slurm_ressources += " --threads-per-core=1"

    slurm_cmd = [
        "#!/bin/bash",
        "#SBATCH --account=cin4181",
        "#SBATCH --constraint=MI250",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --gpus-per-node={args.gpus_per_node}",
        f"#SBATCH --ntasks={args.nodes * args.gpus_per_node}",
        f"#SBATCH --ntasks-per-node={args.gpus_per_node}",
        
        # Adastra have 1 AMD Trento EPYC 7A53 64 cores processors for one node (4GPU)
        # Distribute 64 cores evenly across 4 tasks/GPUs
        f"#SBATCH --cpus-per-task=16", 
        
        # /!\ Caution, 'multithread' in Slurm vocabulary refers to hyperthreading.
        # see https://dci.dci-gitlab.cines.fr/webextranet/user_support/index.html#shared-mode-vs-exclusive-mode
        # 1 process per physical core, no hyperthreading, per default
        "#SBATCH --hint=nomultithread" if args.allow_hyper_threading else '', 
        
        f"#SBATCH --time={args.wall_time}:00:00",
        # name of output and error files
        f"#SBATCH --output={WORK_DIR}/launched_slurm_cmds/{args.run_name}_%j.out",
        f"#SBATCH --error={WORK_DIR}/launched_slurm_cmds/{args.run_name}_%j.out ",
        
        "module purge", # cleans out the modules loaded in interactive and inherited by default
        f"cd {REPO_DIR}",
        f"source ./scripts/activate_world_model_env.sh"
        "pip install .",
        
        "export MPICH_GPU_SUPPORT_ENABLED=1",

        'echo "SLURM_JOBID="$SLURM_JOBID',
        'echo "SLURM_PROCID="$SLURM_PROCID',
        "export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))",
        'echo "MASTER_PORT="$MASTER_PORT',
        'master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)',
        "export MASTER_ADDR=$master_addr",
        'echo "MASTER_ADDR="$MASTER_ADDR',
        "export MIOPEN_USER_DB_PATH=/tmp/miopen_${MASTER_ADDR}_${SLURM_PROCID}",
        'echo "MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH',
        "export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen_${MASTER_ADDR}_${SLURM_PROCID}_cache/",
        'echo "MIOPEN_CUSTOM_CACHE_DIR"=$MIOPEN_CUSTOM_CACHE_DIR',

        "export HYDRA_FULL_ERROR=1",
        
        "# echo of launched commands",
        "set -x",
        
        f'srun {slurm_ressources} python ./world_model/train.py  {devices_args} {args.python_cmd}  name={args.run_name}',
    ]

    slurm_cmd = "\n".join(slurm_cmd)

    slurm_cmd_file = f"{WORK_DIR}/slurm_cmds/{args.run_name}_slurm_cmd.sh"

    with open(slurm_cmd_file, mode="w") as file:
        file.write(slurm_cmd)

    os.system(f"sbatch {slurm_cmd_file}")