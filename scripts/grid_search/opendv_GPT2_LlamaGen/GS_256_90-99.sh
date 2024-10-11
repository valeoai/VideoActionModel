#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 4 -n GS256090_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0341_lr0.0170 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0341 optimizer.lr=0.0170 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256091_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1164_lr0.0587 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1164 optimizer.lr=0.0587 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256092_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0378_lr0.0051 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0378 optimizer.lr=0.0051 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256093_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0772_lr0.0005 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0772 optimizer.lr=0.0005 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256094_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1706_lr0.0015 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1706 optimizer.lr=0.0015 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256095_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0488_lr0.0009 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0488 optimizer.lr=0.0009 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256096_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0266_lr0.0158 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0266 optimizer.lr=0.0158 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256097_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1769_lr0.0002 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1769 optimizer.lr=0.0002 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256098_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.0159_lr0.0004 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0159 optimizer.lr=0.0004 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 4 -n GS256099_muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim256_std0.1604_lr0.0001 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1604 optimizer.lr=0.0001 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

