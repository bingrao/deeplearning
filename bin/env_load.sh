#!/bin/bash
# Root envs
export RootPath=`pwd`
export PYTHONPATH=${PYTHONPATH}:${RootPath}
RootSrc=${RootPath}/nmt
RootData=${RootPath}/data
CurrentDate=$(date +%F)
# load conda virtual environment

conda activate transformer_pytorch

# load cuda 10.1
module load cuda/10.1