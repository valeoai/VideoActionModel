#!/bin/bash

printf "\033[0;32m Launching isort \033[0m\n"
isort world_model scripts inference ./*.py

printf "\033[0;32m Launching black \033[0m\n"
black world_model scripts inference ./*.py

printf "\033[0;32m Launching flake8 \033[0m\n"
flake8 world_model scripts inference ./*.py
