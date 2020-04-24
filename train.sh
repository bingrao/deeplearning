#!/bin/bash
export Project_Home=$(pwd)
export Data_Path=${Project_Home}\data
python train.py --data_dir=data/example/processed --save_config=checkpoints/example_config.json --save_checkpoint=checkpoints/example_model.pth --save_log=logs/example.log
