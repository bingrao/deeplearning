#!/bin/bash
export Project_Home=`pwd`
export Data_Path=${Project_Home}\data

python evaluate.py --save_result=logs/example_eval.txt --config=checkpoints/example_config.json --checkpoint=checkpoints/example_model.pth