#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS128000_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0228_lr0.0335 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0228 optimizer.lr=0.0335 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128001_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0523_lr0.0022 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0523 optimizer.lr=0.0022 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128002_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0678_lr0.0013 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0678 optimizer.lr=0.0013 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128003_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0328_lr0.0045 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0328 optimizer.lr=0.0045 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128004_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.1525_lr0.0048 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1525 optimizer.lr=0.0048 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128005_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0274_lr0.0011 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0274 optimizer.lr=0.0011 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128006_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.3530_lr0.0902 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.3530 optimizer.lr=0.0902 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128007_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.2245_lr0.0417 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.2245 optimizer.lr=0.0417 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128008_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0218_lr0.0032 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0218 optimizer.lr=0.0032 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128009_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0465_lr0.0116 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0465 optimizer.lr=0.0116 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

