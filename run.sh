#!/bin/bash

if [ "$#" -ne 1 ] ; then
  echo "Missing Parameters ..."
  echo "Usage: $0 model[dataset|train|predict|val]" >&2
  exit 1
fi
model=$1

# Root envs
export RootPath=`pwd`
export PYTHONPATH=${PYTHONPATH}:${RootPath}
RootSrc=${RootPath}/nmt
RootData=${RootPath}/data
CurrentDate=$(date +%F)


# Project envs
ProjectName="example"
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
      python "${RootSrc}"/train/train_example_gpu.py \
                                      --project_name="${ProjectName}" \
                                      --project_raw_dir="${ProjectRawDataDir}" \
                                      --project_processed_dir="${ProjectProcessedDataDir}" \
                                      --project_config="${ProjectConfig}" \
                                      --project_log="${ProjectLog}" \
                                      --project_checkpoint="${ProjectCheckpoint}"
  ;;
  "predict")
      set -x
      python "${RootSrc}"/predict/predict.py \
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
      python "${RootSrc}"/evaluate/evaluate.py \
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
     echo "Usage: $0 model[dataset|model|predict|valuate]" >&2
     exit 1
  ;;
esac
