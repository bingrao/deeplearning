#!/bin/bash
export Project_HOME=`pwd`
# load conda virtual environment

conda activate transformer_pytorch

# load cuda 10.1
module load cuda/10.1