#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 4 -n GS256020_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0538_lr0.0006 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0538 optimizer.lr=0.0006 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256021_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1567_lr0.0008 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1567 optimizer.lr=0.0008 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256022_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0496_lr0.0229 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0496 optimizer.lr=0.0229 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256023_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0111_lr0.0003 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0111 optimizer.lr=0.0003 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256024_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.2490_lr0.0049 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.2490 optimizer.lr=0.0049 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256025_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0112_lr0.0006 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0112 optimizer.lr=0.0006 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256026_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0165_lr0.0728 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0165 optimizer.lr=0.0728 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256027_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1493_lr0.0007 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1493 optimizer.lr=0.0007 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256028_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.2429_lr0.0032 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.2429 optimizer.lr=0.0032 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256029_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0118_lr0.0189 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0118 optimizer.lr=0.0189 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

