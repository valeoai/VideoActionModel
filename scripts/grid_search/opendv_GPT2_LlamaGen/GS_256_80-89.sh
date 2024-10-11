#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 4 -n GS256080_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0235_lr0.0012 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0235 optimizer.lr=0.0012 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256081_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.2248_lr0.0054 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.2248 optimizer.lr=0.0054 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256082_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0347_lr0.0013 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0347 optimizer.lr=0.0013 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256083_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1193_lr0.0010 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1193 optimizer.lr=0.0010 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256084_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0844_lr0.0013 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0844 optimizer.lr=0.0013 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256085_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0330_lr0.0003 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0330 optimizer.lr=0.0003 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256086_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0798_lr0.0007 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0798 optimizer.lr=0.0007 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256087_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0411_lr0.0229 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0411 optimizer.lr=0.0229 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256088_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.5010_lr0.0002 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.5010 optimizer.lr=0.0002 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256089_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1324_lr0.0546 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1324 optimizer.lr=0.0546 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

