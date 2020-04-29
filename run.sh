#!/bin/bash

if [ "$#" -ne 2 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 project[dummy|example|spacy] model[preprocess|train|predict|val] " >&2
  exit 1
fi
#ProjectName="spacy"
#ProjectName="dummy"
#ProjectName="example"
ProjectName=$1
model=$2

# Root envs
export RootPath=`pwd`
export PYTHONPATH=${PYTHONPATH}:${RootPath}
CurrentDate=$(date +%F)
ProjectBechmarks=${RootPath}/benchmarks/${ProjectName}
ProjectData=${RootPath}/data/${ProjectName}

# Project envs
ProjectRawDataDir=${ProjectData}/raw
ProjectProcessedDataDir=${ProjectData}/processed
ProjectConfig=${ProjectData}/configs/${ProjectName}_config.json
ProjectLog=${ProjectData}/logs/${ProjectName}-${model}.log
ProjectCheckpoint=${ProjectData}/checkpoints/${CurrentDate}-${ProjectName}-model.pth

case ${model} in
  "preprocess")
      set -x
      python "${ProjectBechmarks}"/preprocess.py \
                              --project_name="${ProjectName}" \
                              --project_raw_dir="${ProjectRawDataDir}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}"
  ;;
  "train")
      set -x
      python "${ProjectBechmarks}"/train.py \
                                      --project_name="${ProjectName}" \
                                      --project_raw_dir="${ProjectRawDataDir}" \
                                      --project_processed_dir="${ProjectProcessedDataDir}" \
                                      --project_config="${ProjectConfig}" \
                                      --project_log="${ProjectLog}" \
                                      --project_checkpoint="${ProjectCheckpoint}" \
                                      --device='cuda'

  ;;
  "predict")
      set -x
      python "${ProjectBechmarks}"/predict.py \
                              --project_name="${ProjectName}" \
                              --project_raw_dir="${ProjectRawDataDir}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --source="There is an imbalance here ."
  ;;
  "val")
      set -x
      python "${ProjectBechmarks}"/evaluate.py \
                              --project_name="${ProjectName}" \
                              --project_raw_dir="${ProjectRawDataDir}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --save_result=logs/example_eval.txt
  ;;
   *)
     echo "There is no match case for ${model}"
     echo "Usage: $0 project[dummy|example|spacy] model[preprocess|train|predict|val] " >&2
     exit 1
  ;;
esac
