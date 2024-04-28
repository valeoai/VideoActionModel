#!/bin/bash 

python adastra_slurm_job_submit.py -n GS128020_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0589_lr0.0743 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0589 optimizer.lr=0.0743 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128021_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.3831_lr0.0612 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.3831 optimizer.lr=0.0612 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128022_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0484_lr0.0011 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0484 optimizer.lr=0.0011 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128023_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0739_lr0.0029 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0739 optimizer.lr=0.0029 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128024_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.3445_lr0.0155 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.3445 optimizer.lr=0.0155 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128025_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0444_lr0.0578 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0444 optimizer.lr=0.0578 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128026_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.4501_lr0.0048 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.4501 optimizer.lr=0.0048 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128027_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0792_lr0.0067 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0792 optimizer.lr=0.0067 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128028_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0148_lr0.0022 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0148 optimizer.lr=0.0022 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128029_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0571_lr0.0176 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0571 optimizer.lr=0.0176 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

