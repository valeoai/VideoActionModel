# Large-Scale PyTorch Model Testing

The goal is to determine the maximum batch size that can be used without encountering out-of-memory (OOM) errors, optimizing the use of computational resources in a high-performance computing (HPC) environment.

This repository contains scripts designed to test various batch sizes for training large-scale PyTorch models distributed across multiple GPUs and nodes. 

## Repository Contents

- `pytorch_lightning_script.py` - A dummy PyTorch Lightning implementation to run a dummy model across multiple GPUs and nodes using Fully Sharded Data Parallel (FSDP).
- `wrapper_script.py` - A script that wraps any given PyTorch training script, that captures output related to memory errors.
- `slurm_script.sh` - A Slurm job submission script that specifies resource requirements and manages the execution of the PyTorch scripts across a distributed system.
- `README.md` - This file, which provides an overview of the repository and instructions for running the scripts.