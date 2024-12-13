from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as open_file:
    install_requires = open_file.read()

with open("dev_requirements.txt") as open_file:
    install_requires_dev = open_file.read()

setup(
    name="world_model",
    version="0.0.1",
    description="",
    author="",
    author_email="florent.bartoccioni@valeo.com",
    url="https://github.com/valeoai/NextTokenPredictor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: TO DEFINE",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
    install_requires=install_requires,
    dependency_links=["https://download.pytorch.org/whl/nightly/cpu"],
    extras_require={
        "dev": install_requires_dev,
        "torch": ["torch==2.4.0", "torchvision==0.19.0"],
    },
)
