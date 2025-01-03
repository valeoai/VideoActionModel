# World model

## Install

To use the world model, you need to install the following dependencies:

```bash
git clone https://www.github.com/valeoai/NextTokenPredictor
cd NextTokenPredictor
pip install -e .
# to install torch at the same time: pip install -e ".[torch]"
```

### Install hyperqueue

```bash
wget https://github.com/It4innovations/hyperqueue/releases/download/v0.20.0/hq-v0.20.0-linux-x64.tar.gz
mkdir -p $WORK/bin
tar -C $WORK/bin -xvzf hq-v0.20.0-linux-x64.tar.gz
```

## DATA

Follow the instructions in the [opendv](opendv/README.md) folder.

## Training
