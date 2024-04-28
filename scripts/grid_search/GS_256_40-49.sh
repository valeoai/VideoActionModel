#!/bin/bash 

python adastra_slurm_job_submit.py -n GS256040_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.2320_lr0.0037 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.2320 optimizer.lr=0.0037 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256041_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.4266_lr0.0723 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.4266 optimizer.lr=0.0723 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256042_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1007_lr0.0035 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1007 optimizer.lr=0.0035 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256043_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.3209_lr0.0076 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.3209 optimizer.lr=0.0076 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256044_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.0119_lr0.0485 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.0119 optimizer.lr=0.0485 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256045_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1759_lr0.0332 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1759 optimizer.lr=0.0332 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256046_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1416_lr0.0018 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1416 optimizer.lr=0.0018 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256047_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.4417_lr0.0469 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.4417 optimizer.lr=0.0469 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256048_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1287_lr0.0029 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1287 optimizer.lr=0.0029 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

python adastra_slurm_job_submit.py -n GS256049_muP_GPT2_Nodes1_BSperGPU32_totalBS256_dim256_std0.1408_lr0.0016 --gpus_per_node 8 --nodes 1  -p 'experiment=muP_GPT2_vqgan_imagenet_f16_1024 model.network.init_std=0.1408 optimizer.lr=0.0016 model.network.embedding_dim=256 model.network.nb_heads=4 data.dataloader_params.batch_size=32 paths.output_dir=/lus/work/CT10/cin4181/SHARED/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 data.dataloader_params.num_workers=2'

sleep 1

