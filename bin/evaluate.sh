#!/bin/bash
export Project_Home=`pwd`/..
export Data_Dir=${Project_Home}/data/example
export Src_Dir=${Project_Home}/src
export Config_Path=${Project_Home}/configs/example_config.json
Data_Raw=${Data_Dir}/raw
Data_Processed=${Data_Dir}/processed
Data_Checkpoint=${Data_Dir}/checkpoints
src_train_path=${Data_Raw}/src-train.txt
tgt_train_path=${Data_Raw}/tgt-train.txt
src_val_path=${Data_Raw}/src-val.txt
tgt_val_path=${Data_Raw}/tgt-val.txt

python ${Src_Dir}/evaluate.py --train_source=${src_train_path} --train_target=${tgt_train_path} --val_source=${src_val_path} --val_target=${tgt_val_path} --save_data_dir=${Data_Processed} --save_result=logs/example_eval.txt --config=checkpoints/example_config.json --checkpoint=checkpoints/example_model.pth
