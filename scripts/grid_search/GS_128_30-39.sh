#!/bin/bash 

python adastra_slurm_job_submit.py -n GS128030_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0439_lr0.0243 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0439 optimizer.lr=0.0243 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128031_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0125_lr0.0011 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0125 optimizer.lr=0.0011 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128032_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0379_lr0.0050 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0379 optimizer.lr=0.0050 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128033_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.2252_lr0.0661 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.2252 optimizer.lr=0.0661 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128034_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.4596_lr0.0440 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.4596 optimizer.lr=0.0440 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128035_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.1042_lr0.0025 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1042 optimizer.lr=0.0025 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128036_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0552_lr0.0159 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0552 optimizer.lr=0.0159 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128037_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0552_lr0.0036 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0552 optimizer.lr=0.0036 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128038_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.0162_lr0.0017 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0162 optimizer.lr=0.0017 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS128039_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim128_std0.1057_lr0.0072 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1057 optimizer.lr=0.0072 model.network.embedding_dim=128 model.network.nb_heads=2 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

