#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 4 -n GS256030_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1550_lr0.0020 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1550 optimizer.lr=0.0020 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256031_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0347_lr0.0005 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0347 optimizer.lr=0.0005 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256032_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1960_lr0.0045 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1960 optimizer.lr=0.0045 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256033_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0244_lr0.0003 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0244 optimizer.lr=0.0003 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256034_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0661_lr0.0190 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0661 optimizer.lr=0.0190 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256035_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0555_lr0.0958 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0555 optimizer.lr=0.0958 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256036_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0549_lr0.0006 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0549 optimizer.lr=0.0006 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256037_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0771_lr0.0197 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0771 optimizer.lr=0.0197 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256038_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.4338_lr0.0002 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.4338 optimizer.lr=0.0002 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256039_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.4223_lr0.0236 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.4223 optimizer.lr=0.0236 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

