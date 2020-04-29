#!/bin/bash

if [ "$#" -ne 2 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 project[dummy|example|spacy] model[preprocess|train|predict|val] " >&2
  exit 1
fi
CurrentDate=$(date +%F)

#ProjectName="spacy"
#ProjectName="dummy"
#ProjectName="example"
ProjectName=$1

# preprocess|train|predict|val
model=$2

# Root envs
export RootPath=`pwd`
export PYTHONPATH=${PYTHONPATH}:${RootPath}

ProjectBechmarks=${RootPath}/benchmarks/${ProjectName}


# Project envs
ProjectData=${ProjectBechmarks}/data

# Processed data folder to save intermediate data
ProjectProcessedDataDir=${ProjectBechmarks}/processed

# Default config file
ProjectConfig=${ProjectBechmarks}/configs/default_config.json

# Default logs file
ProjectLog=${ProjectBechmarks}/logs/${model}-${CurrentDate}.log

# Default Checkpoint file
ProjectCheckpoint=${ProjectBechmarks}/checkpoints/checkpoint-${model}-${CurrentDate}.pth

case ${model} in
  "preprocess")
      set -x
      python "${ProjectBechmarks}"/preprocess.py \
                              --project_name="${ProjectName}" \
                              --project_config="${ProjectConfig}" \
                              --project_raw_dir="${ProjectData}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --phase="${model}"
  ;;
  "train")
      set -x
      python "${ProjectBechmarks}"/train.py \
                                      --project_name="${ProjectName}" \
                                      --project_config="${ProjectConfig}" \
                                      --project_raw_dir="${ProjectData}" \
                                      --project_processed_dir="${ProjectProcessedDataDir}" \
                                      --project_config="${ProjectConfig}" \
                                      --project_checkpoint="${ProjectCheckpoint}" \
                                      --phase="${model}" \
                                      --device='cuda'

  ;;
  "predict")
      set -x
      python "${ProjectBechmarks}"/predict.py \
                              --project_name="${ProjectName}" \
                              --project_config="${ProjectConfig}" \
                              --project_raw_dir="${ProjectData}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --phase="${model}" \
                              --source="There is an imbalance here ."
  ;;
  "val")
      set -x
      python "${ProjectBechmarks}"/evaluate.py \
                              --project_name="${ProjectName}" \
                              --project_config="${ProjectConfig}" \
                              --project_raw_dir="${ProjectData}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --phase="${model}" \
                              --save_result="${ProjectProcessedDataDir}"/${ProjectName}_eval.txt
  ;;
   *)
     echo "There is no match case for ${model}"
     echo "Usage: $0 project[dummy|example|spacy] model[preprocess|train|predict|val] " >&2
     exit 1
  ;;
esac
