#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS128060_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0497_lr0.0024 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0497 optimizer.lr=0.0024 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128061_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0287_lr0.0314 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0287 optimizer.lr=0.0314 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128062_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.2976_lr0.0792 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.2976 optimizer.lr=0.0792 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128063_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0247_lr0.0477 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0247 optimizer.lr=0.0477 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128064_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0580_lr0.0946 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0580 optimizer.lr=0.0946 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128065_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0419_lr0.0012 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0419 optimizer.lr=0.0012 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128066_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0101_lr0.0297 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0101 optimizer.lr=0.0297 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128067_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.1685_lr0.0253 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1685 optimizer.lr=0.0253 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128068_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0108_lr0.0274 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0108 optimizer.lr=0.0274 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128069_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0386_lr0.0291 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0386 optimizer.lr=0.0291 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

