#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 4 -n GS256000_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1302_lr0.0022 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1302 optimizer.lr=0.0022 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256001_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.3373_lr0.0181 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.3373 optimizer.lr=0.0181 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256002_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0145_lr0.0033 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0145 optimizer.lr=0.0033 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256003_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1813_lr0.0008 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1813 optimizer.lr=0.0008 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256004_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.2784_lr0.0095 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.2784 optimizer.lr=0.0095 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256005_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0260_lr0.0010 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0260 optimizer.lr=0.0010 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256006_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0160_lr0.0005 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0160 optimizer.lr=0.0005 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256007_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1341_lr0.0008 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1341 optimizer.lr=0.0008 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256008_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0165_lr0.0348 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0165 optimizer.lr=0.0348 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256009_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0105_lr0.0554 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0105 optimizer.lr=0.0554 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

