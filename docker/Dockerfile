FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y git

RUN pip install --upgrade pip

RUN pip install torchvision==0.19.0
RUN pip install lightning
RUN pip install hydra-core==1.3.2
RUN pip install hydra-colorlog==1.2.0
RUN pip install rich
RUN pip install tqdm
RUN pip install mup
RUN pip install deepspeed
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install pydantic
RUN pip install einops
RUN pip install gitpython
RUN pip install matplotlib
RUN pip install --no-dependencies nuscenes-devkit
RUN pip install opencv-python
RUN pip install scikit-learn
RUN pip install pyquaternion
RUN pip install cachetools
RUN pip install descartes
RUN pip install fire
RUN pip install "Shapely<2.0.0"
RUN pip install parameterized

ADD . /NextTokenPredictor
RUN cd /NextTokenPredictor && pip install --no-cache-dir .

# Bash as default entrypoint.
CMD /bin/bash -l
