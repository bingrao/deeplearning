#!/bin/bash
export Project_Home=`pwd`
export Data_Path=${Project_Home}\data
python datasets.py --train_source=data/example/raw/src-train.txt --train_target=data/example/raw/tgt-train.txt --val_source=data/example/raw/src-val.txt --val_target=data/example/raw/tgt-val.txt --save_data_dir=data/example/processed