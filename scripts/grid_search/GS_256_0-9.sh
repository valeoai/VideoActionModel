#!/bin/bash 

python adastra_slurm_job_submit.py -n GS256000_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0229_lr0.0974 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0229 optimizer.lr=0.0974 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256001_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1098_lr0.0209 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1098 optimizer.lr=0.0209 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256002_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0526_lr0.0064 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0526 optimizer.lr=0.0064 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256003_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0852_lr0.0512 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0852 optimizer.lr=0.0512 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256004_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1342_lr0.0725 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1342 optimizer.lr=0.0725 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256005_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1886_lr0.0085 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1886 optimizer.lr=0.0085 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256006_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0849_lr0.0025 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0849 optimizer.lr=0.0025 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256007_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0970_lr0.0019 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0970 optimizer.lr=0.0019 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256008_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0158_lr0.0389 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0158 optimizer.lr=0.0389 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256009_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.2686_lr0.0056 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.2686 optimizer.lr=0.0056 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

