#!/bin/bash --login
set -e

conda activate $ENV_PREFIX
python unires/fit_unires.py "$@"
