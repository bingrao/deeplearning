#!/bin/bash
export Project_Home=`pwd`
export Data_Path=${Project_Home}\data
# load conda virtual environment

conda activate transformer_pytorch

# load cuda 10.1
module load cuda/10.1