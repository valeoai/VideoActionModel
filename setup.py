#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="world_model",
    version="0.0.1",
    description="",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "world_model_train = world_model.train:main",
            "world_model_eval = world_model.eval:main",
        ]
    },
)
