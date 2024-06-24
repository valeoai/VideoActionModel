#!/bin/bash 

python adastra_slurm_job_submit.py -n GS256030_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0439_lr0.0243 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0439 optimizer.lr=0.0243 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256031_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0125_lr0.0011 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0125 optimizer.lr=0.0011 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256032_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0379_lr0.0050 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0379 optimizer.lr=0.0050 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256033_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.2252_lr0.0661 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.2252 optimizer.lr=0.0661 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256034_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.4596_lr0.0440 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.4596 optimizer.lr=0.0440 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256035_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1042_lr0.0025 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1042 optimizer.lr=0.0025 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256036_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0552_lr0.0159 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0552 optimizer.lr=0.0159 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256037_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0552_lr0.0036 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0552 optimizer.lr=0.0036 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256038_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0162_lr0.0017 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0162 optimizer.lr=0.0017 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256039_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1057_lr0.0072 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1057 optimizer.lr=0.0072 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

