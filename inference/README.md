# Neuro-NCAP evaluation

## Local install part

Tutorial to install NeuroNCAP. This is the first part that must be done on the local cluster to have access to docker.

First define the base directory where to store all folder:

```bash
export BASE_REPO=~/shared/eramzi/
cd $BASE_REPO
```

Clone all the repository.

```bash
git clone https://github.com/atonderski/neuro-ncap.git
git clone https://github.com/georghess/neurad-studio.git
# We assume that the NextTokenPredictor repo is already here
```

Build the different docker images and save them as tar file

NeuroNCap:

```bash
cd $BASE_REPO/neuro-ncap
docker build -t ncap:latest -f docker/Dockerfile .
docker save -o ncap.docker.tar.gz ncap:latest
```

Neurad-studio:

/!\ To make it work on Jean-Zay (V100) I had to add `70` to the `CUDA_ARCHITECTURES` in the `Dockerfile` of `neurad-studio`.

```bash
cd $BASE_REPO/neurad-studio
docker build -t neurad:latest -f Dockerfile
docker save -o neurad.docker.tar.gz neurad:latest
```

Video Action Model:

```bash
cd $BASE_REPO/NextTokenPredictor
docker build -t ncap_vam:latest -f docker/Dockerfile .
docker save -o ncap_vam.docker.tar.gz ncap_vam:latest
```

Then send them all to JZ.

```bash
export $DOCKER_JZ_FOLDER=$ycy_ALL_CCFRSCRATCH/neuroncap_docker_file  # you need to define this
scp $BASE_REPO/neuro-ncap/ncap.docker.tar.gz jz:$DOCKER_JZ_FOLDER
scp $BASE_REPO/neurad-studio/rendering.docker.tar.gz jz:$DOCKER_JZ_FOLDER
scp $BASE_REPO/NextTokenPredictor/ncap_vam.docker.tar.gz jz:$DOCKER_JZ_FOLDER
```

## Jean-Zay install part

Tutorial to install NeuroNCAP on your SLURM cluster.

First define the base directory where to store all folder:

```bash
export BASE_JZ_REPO=$WORK/
export DOCKER_JZ_FOLDER=$ycy_ALL_CCFRSCRATCH/neuroncap_docker_file
cd $BASE_JZ_REPO
```

Clone the repo on JZ:

```bash
cd $BASE_JZ_REPO
git clone https://github.com/atonderski/neuro-ncap.git
git clone https://github.com/georghess/neurad-studio.git
# We assume that the NextTokenPredictor repo is already here
```

Download the weights / checkpoinst:

```bash
cd $BASE_JZ_REPO/neuro-ncap
bash scripts/download/download_neurad_weights.sh
cd $BASE_JZ_REPO
module purge
module load pytorch-gpu/py3/2.4.0
python -c 'import torchvision; torchvision.models.vgg19(pretrained=True)'
python -c 'import torchvision; torchvision.models.alexnet(pretrained=True)'
```

Then build the singularity images by running the following:

```bash
sbatch scripts/build_singularity_images.slurm
```

### Run the NeuroNCAP pipeline

To run the pipeline, you need to run the following command:

```bash
bash scripts/run_neuro_ncap_eval.sh
```

You can then get the results by running the following command:

```bash
python scripts/evaluate_results.py --result_path /path/to/logs
```

(Optional) You can create videos for qualitative results by running the following command:

```bash
python scripts/create_gif.py --rootdir /path/to/logs
```
