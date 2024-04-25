module purge
source /lus/work/CT10/cin4181/SHARED/shared_python_envs/wm_world_model_env_torch2.4.0dev/bin/activate # Activate custom env
module load craype-accel-amd-gfx90a craype-x86-trento # Compiler ?
module load PrgEnv-cray # devkit ?
module load amd-mixed # AMD hardware