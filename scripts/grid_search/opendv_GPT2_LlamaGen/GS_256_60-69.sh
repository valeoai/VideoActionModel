#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 4 -n GS256060_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0262_lr0.0057 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0262 optimizer.lr=0.0057 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256061_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0135_lr0.0012 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0135 optimizer.lr=0.0012 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256062_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1627_lr0.0001 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1627 optimizer.lr=0.0001 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256063_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0129_lr0.0012 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0129 optimizer.lr=0.0012 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256064_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1462_lr0.0011 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1462 optimizer.lr=0.0011 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256065_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.3980_lr0.0500 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.3980 optimizer.lr=0.0500 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256066_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0107_lr0.0240 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0107 optimizer.lr=0.0240 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256067_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0220_lr0.0012 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0220 optimizer.lr=0.0012 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256068_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0161_lr0.0022 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0161 optimizer.lr=0.0022 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256069_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.3896_lr0.0308 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.3896 optimizer.lr=0.0308 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

