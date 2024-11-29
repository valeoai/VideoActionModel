# Neuro-NCAP evaluation

## Docker image

Build the docker image with:

```bash
docker build -t ncap_wm:latest -f docker/Dockerfile .
```

Save it as a tar file to transfer to Jean-Zaa:

```bash
docker save -o ncap_wm.docker.tar.gz ncap_wm:latest
export $DOCKER_JZ_FOLDER=$ycy_ALL_CCFRSCRATCH/neuroncap_docker_file  # you need to define this
scp ncap_wm.docker.tar.gz jz:$DOCKER_JZ_FOLDER/neuro-ncap
```
