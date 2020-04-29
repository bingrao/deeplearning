#!/bin/bash

if [ "$#" -ne 2 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 project[dummy|example|spacy] model[dataset|train|predict|val] " >&2
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
RootSrc=${RootPath}/nmt
RootData=${RootPath}/data
CurrentDate=$(date +%F)


# Project envs
ProjectData=${RootData}/${ProjectName}
ProjectRawDataDir=${ProjectData}/raw
ProjectProcessedDataDir=${ProjectData}/processed
ProjectConfig=${ProjectData}/configs/${ProjectName}_config.json
ProjectLog=${ProjectData}/logs/${ProjectName}-${model}.log
ProjectCheckpoint=${ProjectData}/checkpoints/${CurrentDate}-${ProjectName}-model.pth

case ${model} in
  "dataset")
      set -x
      python "${RootSrc}"/data/datasets.py \
                              --project_name="${ProjectName}" \
                              --project_raw_dir="${ProjectRawDataDir}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}"
  ;;
  "train")
      set -x
      python "${RootSrc}"/train/train_${ProjectName}.py \
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
      python "${RootSrc}"/predict/predict_${ProjectName}.py \
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
      python "${RootSrc}"/evaluate/evaluate_${ProjectName}.py \
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
     echo "Usage: $0 project[dummy|example|spacy] model[dataset|train|predict|val] " >&2
     exit 1
  ;;
esac
