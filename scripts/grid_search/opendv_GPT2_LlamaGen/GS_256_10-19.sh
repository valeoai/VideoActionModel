#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 8 -n GS256010_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0668_lr0.0017 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0668 optimizer.lr=0.0017 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256011_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0264_lr0.0514 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0264 optimizer.lr=0.0514 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256012_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.3872_lr0.0208 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.3872 optimizer.lr=0.0208 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256013_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.2345_lr0.0002 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.2345 optimizer.lr=0.0002 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256014_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.3278_lr0.0353 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.3278 optimizer.lr=0.0353 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256015_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0980_lr0.0006 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0980 optimizer.lr=0.0006 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256016_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0774_lr0.0080 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0774 optimizer.lr=0.0080 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256017_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0187_lr0.0009 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0187 optimizer.lr=0.0009 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256018_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1963_lr0.0253 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1963 optimizer.lr=0.0253 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256019_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0658_lr0.0005 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0658 optimizer.lr=0.0005 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

