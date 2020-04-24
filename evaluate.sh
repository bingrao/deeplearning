#!/bin/bash
export Project_Home=`pwd`
export Data_Dir=${Project_Home}/data/example
export Src_Dir=${Project_Home}/src
export Config_Path=${Project_Home}/configs/example_config.json

export Data_Raw=${Data_Dir}/raw
export Data_Processed=${Data_Dir}/processed
export Data_Checkpoint=${Data_Dir}/checkpoints

python src/evaluate.py --save_result=logs/example_eval.txt --config=checkpoints/example_config.json --checkpoint=checkpoints/example_model.pth
