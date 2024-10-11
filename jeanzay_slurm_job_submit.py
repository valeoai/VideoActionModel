import os
from argparse import ArgumentParser
from pathlib import Path

from datetime import datetime
import time

SCRATCH_DIR = Path(os.path.expandvars("$SCRATCH"))
WORK_DIR = Path(os.path.expandvars("$WORK"))
REPO_DIR = WORK_DIR / 'NextTokenPredictor'

default_cpu_per_task = 24

if __name__ == "__main__":
    parser = ArgumentParser(prog="PROG")
    parser.add_argument("--run_name", "-n", type=str, required=True)
    parser.add_argument("--python_cmd", "-p", type=str, required=True)
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--qos", type=str, default="t3", choices=["t3", "t4", "dev"])
    parser.add_argument("--account", type=str, default="ycy@h100")
    parser.add_argument("--file_to_run", "-f", type=str, default='train')
    parser.add_argument("--wall_time", "-wt", type=int, default=20) # jean zay has max time of 20h
    parser.add_argument("--allow_hyper_threading", action='store_true')
    # parse params
    args = parser.parse_args()

    print("RUN NAME: ", args.run_name)
    print("ARGS: ", args.python_cmd)

    # Combining these below, the format produces a string like "0418_1530_1650295200", where:
    # "0418" indicates April 18th,
    # "1530" indicates 3:30 PM,
    # "1650295200" indicates the number of seconds since the Unix Epoch 
    # (January 1, 1970 00:00:00 UTC), typically referred to as Unix time.
    current_time = datetime.now()
    time_code = current_time.strftime("%m%d_%H%M_") + str(int(time.mktime(current_time.timetuple())))
    
    run_name = f'{args.run_name}_{time_code}'
    
    devices_args = f'++trainer.devices={args.gpus_per_node} ++trainer.num_nodes={args.nodes}'
    
    qos_full = f"qos_gpu-{args.qos}"

    slurm_cmd = [
        "#!/bin/bash",
        f"#SBATCH --job-name={args.run_name}",
        "#SBATCH -C h100",
        f"#SBATCH --nodes={args.nodes}",
        f"#SBATCH --gres=gpu:{args.gpus_per_node}",
        f"#SBATCH --ntasks-per-node={args.gpus_per_node}",
        f"#SBATCH --cpus-per-task={default_cpu_per_task}", # Logical cores per MPI task  
        
        # nomultithred to get more RAM per GPU
        # see https://github.com/valeoai/VisualQuantization/blob/dev/scripts/SLURM_PREPOST.md
        "#SBATCH --hint=nomultithread" if not args.allow_hyper_threading else '', 
        
        f"#SBATCH --time={args.wall_time}:00:00",
        # name of output and error files
        f"#SBATCH --output={WORK_DIR}/slurm_jobs_logs/stdout/%x_%j.out",
        f"#SBATCH --error={WORK_DIR}/slurm_jobs_logs/stdout/%x_%j.out",
        
        f"#SBATCH --qos={qos_full}",
        f"#SBATCH -A {args.account}",
        
        "module purge", # cleans out the modules loaded in interactive and inherited by default
        "module load arch/h100",
        "module load pytorch-gpu/py3/2.4.0",
        f"export PYTHONUSERBASE={WORK_DIR}/python_envs/worldmodel",
        
        "export MPICH_GPU_SUPPORT_ENABLED=1",
        "NCCL_DEBUG=INFO",
        "CUDA_LAUNCH_BLOCKING=1",
        
        # Important change when using deepspeed (which now uses triton)
        # By default the cache dir will be $HOME/.triton
        # We point it to $SCRATCH because the inodes quota is very limited on JeanZay
        f"export TRITON_CACHE_DIR={SCRATCH_DIR}/.triton",

        "export HYDRA_FULL_ERROR=1",
        
        "# echo of launched commands",
        "set -x",
        
        f'srun python {REPO_DIR}/world_model/{args.file_to_run}.py {args.python_cmd}  {devices_args}  name={run_name}',
    ]

    slurm_cmd = "\n".join(slurm_cmd)

    slurm_cmd_file = f"{WORK_DIR}/slurm_jobs_logs/commands/{run_name}_slurm_cmd.sh"

    assert WORK_DIR.exists()
    (WORK_DIR / 'slurm_jobs_logs/commands').mkdir(parents=True, exist_ok=True)
    (WORK_DIR / 'slurm_jobs_logs/stdout').mkdir(parents=True, exist_ok=True)

    with open(slurm_cmd_file, mode="w") as file:
        file.write(slurm_cmd)

    os.system(f"sbatch {slurm_cmd_file}")