# Preparing OpenDV dataset

## Download the dataset

First follow the instructions in the [OpenDV Dataset](https://github.com/OpenDriveLab/DriveAGI) repository to download the data. And the metadat as a csv file [Metadata](https://docs.google.com/spreadsheets/d/1bHWWP_VXeEe5UzIG-QgKFBdH7mNlSC4GFSJkEhFnt2I).

## Prepare the dataset

Then you need to create the following folder structure:

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

## Tokenizer

Then download the llamagen tokenizer jit file:

```bash
wget https://XXX/llamagen.jit
```

## Preparing the tokens

### Install HQ

Hyperqueue is a tool to scale the data extraction process on HPC clusters.

Hyperqueue uses a servers and several workers to process the data. The server is responsible for managing the workers and the workers are responsible for processing queue of jobs. On SLURM cluster hyperqueue allow to remove the queue time required for each jobs.

Check the [Hyperqueue documentation](https://it4innovations.github.io/hyperqueue/stable/)!

To install the server and the workers, you need to install the hyperqueue package:

```bash
wget https://github.com/It4innovations/hyperqueue/releases/download/v0.20.0/hq-v0.20.0-linux-x64.tar.gz
mkdir -p ~/bin
tar -C ~/bin -xvzf hq-v0.20.0-linux-x64.tar.gz
rm hq-v0.20.0-linux-x64.tar.gz
export PATH=$PATH:~/bin  # you can add this to your .bashrc
pip install hyperqueue==0.20.0
```

### Configure SLURM file

You should configure the HQ SLURM files in `./hq/*.slurm` to match your cluster configuration.

### Extract frames from OpenDV dataset

To extract the frames from the OpenDV dataset, you need to run the following command:

```bash
bash ./scripts/extract_opendv_frames.sh 20 24
```

The first parameter is the number of workers and the second parameter is the number of cpus per worker (this should match the config of your slurm files).

### Tokenize the frames

Once you have the frames extracted, you can tokenize the frames using the following command:

```bash
bash ./scripts/tokenize_opendv_from_frames.sh 10 24
```

It uses the same arguments as the previous command.

You may need to change the batch size in the `./scripts/tokenize_opendv_from_frames.sh` script to match your GPU memory. Also if you use GPUs that are not A100 or H100 you may need to change the dtype (e.g., fp16 instead of bf16).

Finally use the following command to have a flat structure of the tokens:

```bash
python ./scripts/flatten_opendv_tokens.py \
--rootdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/tokens \
--outdir $fzh_ALL_CCFRSCRATCH/OpenDV_processed/flat_tokens
```
