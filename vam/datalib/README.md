# Data

## Preparing OpenDV dataset

### Download the dataset

First follow the instructions in the [OpenDV Dataset](https://github.com/OpenDriveLab/DriveAGI) repository to download the different videos of the OpenDV dataset.

Please find in the [data_files.tar.gz](https://www.github.com/valeoai/VideoActionModel) tar file the `metadata.csv` that we used to pre-process the OpenDV dataset. As detailed in our tech report, some of the videos were discarded.

```bash
tar -xvzf data_files.tar.gz
```

### Prepare the dataset

You should have a folder structure similar to the following:

```bash
OpenDV_Youtube
|––––videos
|----------Driver1
|-----------------video1.mp4
|-----------------video2.webm
|-----------------...
|----------Driver2
|----------...
|––––metadata.csv
```

### Tokenizer

Please download the JIT files of the LlamaGen tokenizer on our repository: [llamagen.jit](https://www.github.com/valeoai/VideoActionModel).

### Preparing the tokens

#### Install HQ

Hyperqueue is a tool to scale the data extraction process on HPC clusters.

Hyperqueue uses a servers and several workers to process data. The server is responsible for managing the workers and dispatching jobs while the workers are responsible for the actual processing. On SLURM cluster hyperqueue allow to remove the SLURM queue time that would be required for individual jobs.

Check the [Hyperqueue documentation](https://it4innovations.github.io/hyperqueue/stable/)!

To install the hyperqueue binary and hyperqueue python library:

```bash
wget https://github.com/It4innovations/hyperqueue/releases/download/v0.19.0/hq-v0.19.0-linux-x64.tar.gz
mkdir -p ~/bin
tar -C ~/bin -xvzf hq-v0.19.0-linux-x64.tar.gz
rm hq-v0.19.0-linux-x64.tar.gz
export PATH=$PATH:~/bin  # you can add this to your .bashrc
pip install hyperqueue==0.19.0
```

Note: we were able to install `hyperqueue` on our SLURM cluster without requiring any privileges.

#### Configure SLURM file

You can configure the HQ SLURM files in `./hq/*.slurm` to match your SLURM cluster configuration.

#### Extract frames from OpenDV dataset

To extract the frames from the OpenDV dataset, run the following command:

```bash
bash ./scripts/extract_opendv_frames.sh 20 24
```

The first parameter is the number of workers (each worker will be a SLURM job) and the second parameter is the number of CPUs per worker (this should match the config of your SLURM files). The more worker you have the faster the extraction will be.

#### Tokenize the frames

Once the frames have been extracted, you can tokenize the frames using the following command:

```bash
bash ./scripts/tokenize_opendv_from_frames.sh 10 24
```

It uses the same arguments as the previous command, although the workers now require access to GPUs.

You may need to change the batch size in the `./scripts/tokenize_opendv_from_frames.sh` script to match your GPU memory. Also, if you use GPUs that are not A100s or H100s you may need to change the dtype (e.g., fp16 instead of bf16).

Finally use the following command to create a flat structure of the tokens:

```bash
python ./scripts/flatten_opendv_tokens.py \
--rootdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/tokens \
--outdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens
```

## Preparing nuPlan and nuScenes datasets

Please follow the instructions to download [nuPlan](https://www.nuscenes.org/nuplan) and [nuScenes](https://www.nuscenes.org/) datasets.

After extracting the datasets, you can use the following scripts to prepare the data.

For nuPlan:

```bash
bash ./scripts/tokenize_nuplan.sh 10 24
```

For nuScenes:

```bash
bash ./scripts/tokenize_nuscenes.sh 10 24
```

The scripts are similar to the ones used for the OpenDV dataset.

Finally, you can use the pickle files from [data_files.tar.gz](https://www.github.com/valeoai/VideoActionModel) to use the datasets from our repository.
