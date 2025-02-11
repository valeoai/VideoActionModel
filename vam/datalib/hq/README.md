# HQ folder

This folder contains helper script to handle hyperqueue on a SLURM cluster.

```bash
hq
|––––start_hq_archive_slurm.sh  # Starts the HQ server and workers without GPUs
|––––start_hq_server.sh  # Starts the HQ server on a SLURM compute node
|––––start_hq_slurm.sh  # Starts the HQ server and workers with a single GPU
|––––start_hq_worker_archive.sh  # Starts a single HQ worker without GPUs
|––––start_hq_worker.sh  # Starts a single HQ worker with GPU
```
