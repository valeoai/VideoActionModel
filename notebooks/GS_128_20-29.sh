#!/bin/bash 

python jeanzay_slurm_job_submit.py -n GS128020_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0829_lr0.0449 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0829 optimizer.lr=0.0449 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128021_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.2391_lr0.0053 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.2391 optimizer.lr=0.0053 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128022_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0141_lr0.0027 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0141 optimizer.lr=0.0027 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128023_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0233_lr0.0021 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0233 optimizer.lr=0.0021 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128024_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0450_lr0.0638 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0450 optimizer.lr=0.0638 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128025_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0379_lr0.0034 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0379 optimizer.lr=0.0034 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128026_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0124_lr0.0356 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0124 optimizer.lr=0.0356 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128027_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0563_lr0.0182 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0563 optimizer.lr=0.0182 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128028_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.1049_lr0.0024 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.1049 optimizer.lr=0.0024 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n GS128029_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.4724_lr0.0047 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.4724 optimizer.lr=0.0047 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/next_token_predictor_grid_search ++trainer.max_epochs=1 optimizer.weight_decay=0.0 data.dataloader_params.num_workers=4'

sleep 1

