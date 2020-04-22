#!/bin/bash
export Project_Home=`pwd`
export Data_Path=${Project_Home}\data
python predict.py --source="There is an imbalance here ." --config=checkpoints/example_config.json --checkpoint=checkpoints/example_model.pth