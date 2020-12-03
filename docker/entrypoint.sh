#!/bin/bash --login
set -e

conda activate $ENV_PREFIX
python unires/unires.py "$@"
