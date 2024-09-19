#!/bin/bash 

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0217_lr0.00001 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0000 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0217_lr0.00010 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0001 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0217_lr0.00100 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0010 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes4_BSperGPU24_totalBS384_dim128_std0.0217_lr0.01000 --gpus_per_node 4 --nodes 4  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0100 model.network.embedding_dim=128 model.network.nb_heads=1 data.dataloader_params.batch_size=24 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0217_lr0.00001 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0000 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0217_lr0.00010 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0001 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0217_lr0.00100 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0010 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes6_BSperGPU16_totalBS384_dim256_std0.0217_lr0.01000 --gpus_per_node 4 --nodes 6  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0100 model.network.embedding_dim=256 model.network.nb_heads=2 data.dataloader_params.batch_size=16 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1024_std0.0217_lr0.00001 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0000 model.network.embedding_dim=1024 model.network.nb_heads=8 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1024_std0.0217_lr0.00010 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0001 model.network.embedding_dim=1024 model.network.nb_heads=8 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1024_std0.0217_lr0.00100 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0010 model.network.embedding_dim=1024 model.network.nb_heads=8 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1024_std0.0217_lr0.01000 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0100 model.network.embedding_dim=1024 model.network.nb_heads=8 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1408_std0.0217_lr0.00001 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0000 model.network.embedding_dim=1408 model.network.nb_heads=11 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1408_std0.0217_lr0.00010 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0001 model.network.embedding_dim=1408 model.network.nb_heads=11 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1408_std0.0217_lr0.00100 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0010 model.network.embedding_dim=1408 model.network.nb_heads=11 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1

python jeanzay_slurm_job_submit.py -n muP_GPT2_Nodes24_BSperGPU4_totalBS384_dim1408_std0.0217_lr0.01000 --gpus_per_node 4 --nodes 24  -p 'experiment=muP_GPT2_VQ_ds16_16384_llamagen_nuplan_noaction model.network.init_std=0.0217 optimizer.lr=0.0100 model.network.embedding_dim=1408 model.network.nb_heads=11 data.dataloader_params.batch_size=4 paths.output_dir=/lustre/fsn1/projects/rech/ycy/commun/output_data/nuplan_GPT2_LlamaGen/lr_transfer_validation ++trainer.max_epochs=1 optimizer.weight_decay=0.1 data.dataloader_params.num_workers=4'

sleep 1