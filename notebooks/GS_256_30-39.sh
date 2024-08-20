#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS256030_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1250_lr0.0982 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1250 optimizer.lr=0.0982 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256031_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0245_lr0.0480 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0245 optimizer.lr=0.0480 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256032_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1091_lr0.0195 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1091 optimizer.lr=0.0195 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256033_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0790_lr0.0014 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0790 optimizer.lr=0.0014 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256034_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.3720_lr0.0012 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3720 optimizer.lr=0.0012 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256035_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0424_lr0.0226 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0424 optimizer.lr=0.0226 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256036_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0434_lr0.0034 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0434 optimizer.lr=0.0034 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256037_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1258_lr0.0500 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1258 optimizer.lr=0.0500 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256038_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.2534_lr0.0103 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.2534 optimizer.lr=0.0103 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256039_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0619_lr0.0199 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0619 optimizer.lr=0.0199 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

