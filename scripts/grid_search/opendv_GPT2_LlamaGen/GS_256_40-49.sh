#!/bin/bash 

python jeanzay_slurm_job_submit.py -wt 8 -n GS256040_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0118_lr0.0183 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0118 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0183 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256041_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0194_lr0.0142 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0194 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0142 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256042_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0161_lr0.0095 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0161 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0095 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256043_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0249_lr0.0039 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0249 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0039 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256044_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0124_lr0.0088 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0124 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0088 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256045_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0155_lr0.0284 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0155 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0284 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256046_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0928_lr0.0035 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0928 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0035 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256047_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.2148_lr0.0129 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.2148 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0129 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256048_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0741_lr0.0648 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0741 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0648 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

python jeanzay_slurm_job_submit.py -wt 8 -n GS256049_muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0397_lr0.0282 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_opendv_noaction model.network.init_std=0.0397 callbacks=callbacks_opendv_grid_search optimizer.lr=0.0282 model.network.embedding_dim=256 model.network.nb_heads=2 data.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/opendv_GPT2_LlamaGen/grid_search_2 ++trainer.max_epochs=1 ++trainer.limit_train_batches=14984 optimizer.weight_decay=0.0 data.num_workers=6'

sleep 1

