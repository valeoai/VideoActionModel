#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 8 -n GS256010_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0305_lr0.0025 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0305 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0025 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256011_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0568_lr0.0356 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0568 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0356 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256012_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0669_lr0.0645 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0669 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0645 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256013_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0340_lr0.0744 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0340 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0744 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256014_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0409_lr0.0087 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0409 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0087 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256015_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0361_lr0.0900 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0361 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0900 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256016_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1777_lr0.0459 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1777 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0459 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256017_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.1404_lr0.0194 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.1404 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0194 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256018_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.3179_lr0.0048 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.3179 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0048 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256019_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0260_lr0.0546 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0260 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0546 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

