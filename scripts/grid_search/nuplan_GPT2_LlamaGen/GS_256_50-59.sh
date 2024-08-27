#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS256050_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.3762_lr0.0051 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3762 optimizer.lr=0.0051 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256051_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0463_lr0.0954 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0463 optimizer.lr=0.0954 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256052_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.3558_lr0.0011 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3558 optimizer.lr=0.0011 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256053_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0704_lr0.0469 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0704 optimizer.lr=0.0469 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256054_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0163_lr0.0058 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0163 optimizer.lr=0.0058 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256055_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0120_lr0.0001 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0120 optimizer.lr=0.0001 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256056_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0196_lr0.0034 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0196 optimizer.lr=0.0034 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256057_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.2788_lr0.0006 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.2788 optimizer.lr=0.0006 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256058_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0994_lr0.0314 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0994 optimizer.lr=0.0314 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS256059_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0332_lr0.0010 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0332 optimizer.lr=0.0010 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

