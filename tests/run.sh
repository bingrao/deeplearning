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

#if [ ! -f ${LOG_FILE} ]
#then
#  echo "Log file does not exist and create it ..."
#  mkdir -p ${APP_HOME}/log
#  touch ${LOG_FILE}
#fi


# Will deprecate in future ...
src_train_path=${ProjectRawDataDir}/src-train.txt
tgt_train_path=${ProjectRawDataDir}/tgt-train.txt
src_val_path=${ProjectRawDataDir}/src-val.txt
tgt_val_path=${ProjectRawDataDir}/tgt-val.txt


case ${model} in
  "dataset")
      set -x
      python "${RootSrc}"/data/datasets.py \
                              --project_name="${ProjectName}" \
                              --project_raw_dir="${ProjectRawDataDir}" \
                              --project_processed_dir="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --train_source="${src_train_path}" \
                              --train_target="${tgt_train_path}" \
                              --val_source="${src_val_path}" \
                              --val_target="${tgt_val_path}" \
                              --save_data_dir="${ProjectProcessedDataDir}" \
                              --save_log="${ProjectLog}"
  ;;
  "train")
      set -x
      python "${RootSrc}"/train/train.py \
                                      --project_name="${ProjectName}" \
                                      --project_raw_data="${ProjectRawDataDir}" \
                                      --project_processed_data="${ProjectProcessedDataDir}" \
                                      --project_config="${ProjectConfig}" \
                                      --project_log="${ProjectLog}" \
                                      --project_checkpoint="${ProjectCheckpoint}" \
                                      --train_source="${src_train_path}" \
                                      --train_target="${tgt_train_path}" \
                                      --val_source="${src_val_path}" \
                                      --val_target="${tgt_val_path}" \
                                      --save_data_dir="${ProjectProcessedDataDir}" \
                                      --save_config="${ProjectData}" \
                                      --checkpoint="${ProjectCheckpoint}" \
                                      --save_log="${ProjectLog}"
  ;;
  "predict")
      set -x
      python "${RootSrc}"/predict/predict.py \
                              --project_name="${ProjectName}" \
                              --project_raw_data="${ProjectRawDataDir}" \
                              --project_processed_data="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --train_source="${src_train_path}" \
                              --train_target="${tgt_train_path}" \
                              --val_source="${src_val_path}" \
                              --val_target="${tgt_val_path}" \
                              --save_data_dir="${ProjectProcessedDataDir}" \
                              --source="There is an imbalance here ." \
                              --checkpoint="${ProjectCheckpoint}" \
                              --save_log="${ProjectLog}"
  ;;
  "val")
      set -x
      python "${RootSrc}"/evaluate/evaluate.py \
                              --project_name="${ProjectName}" \
                              --project_raw_data="${ProjectRawDataDir}" \
                              --project_processed_data="${ProjectProcessedDataDir}" \
                              --project_config="${ProjectConfig}" \
                              --project_log="${ProjectLog}" \
                              --project_checkpoint="${ProjectCheckpoint}" \
                              --train_source="${src_train_path}" \
                              --train_target="${tgt_train_path}" \
                              --val_source="${src_val_path}" \
                              --val_target="${tgt_val_path}" \
                              --save_data_dir="${ProjectProcessedDataDir}" \
                              --save_result=logs/example_eval.txt \
                              --save_config="${ProjectCheckpoint}" \
                              --checkpoint="${ProjectCheckpoint}" \
                              --save_log="${ProjectLog}"
  ;;
   *)
     echo "There is no match case for ${model}"
     echo "Usage: $0 model[dataset|model|predict|valuate]" >&2
     exit 1
  ;;
esac
