#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS128020_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.1550_lr0.0022 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1550 optimizer.lr=0.0022 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128021_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0157_lr0.0062 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0157 optimizer.lr=0.0062 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128022_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.1008_lr0.0001 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1008 optimizer.lr=0.0001 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128023_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.3195_lr0.0016 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3195 optimizer.lr=0.0016 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128024_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0479_lr0.0165 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0479 optimizer.lr=0.0165 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128025_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0153_lr0.0191 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0153 optimizer.lr=0.0191 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128026_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.4900_lr0.0050 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.4900 optimizer.lr=0.0050 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128027_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0169_lr0.0001 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0169 optimizer.lr=0.0001 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128028_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0638_lr0.0025 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0638 optimizer.lr=0.0025 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128029_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.2945_lr0.0146 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.2945 optimizer.lr=0.0146 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

