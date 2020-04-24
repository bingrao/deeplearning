#!/bin/bash
export Project_Home=`pwd`
export Data_Dir=${Project_Home}/data/example
export Src_Dir=${Project_Home}/src
export Config_Path=${Project_Home}/configs/example_config.json

export Data_Raw=${Data_Dir}/raw
export Data_Processed=${Data_Dir}/processed
export Data_Checkpoint=${Data_Dir}/checkpoints
python train_standalone.py --data_dir=data/example/processed --save_config=checkpoints/example_config.json --save_checkpoint=checkpoints/example_model.pth --save_log=logs/example.log
