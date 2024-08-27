#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS128030_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0743_lr0.0641 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0743 optimizer.lr=0.0641 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128031_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.3339_lr0.0458 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3339 optimizer.lr=0.0458 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128032_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.3806_lr0.0095 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3806 optimizer.lr=0.0095 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128033_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.4181_lr0.0015 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.4181 optimizer.lr=0.0015 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128034_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0706_lr0.0004 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0706 optimizer.lr=0.0004 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128035_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0150_lr0.0421 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0150 optimizer.lr=0.0421 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128036_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0172_lr0.0073 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0172 optimizer.lr=0.0073 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128037_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0231_lr0.0056 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0231 optimizer.lr=0.0056 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128038_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0153_lr0.0121 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0153 optimizer.lr=0.0121 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128039_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.3730_lr0.0316 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3730 optimizer.lr=0.0316 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

