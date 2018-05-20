#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName machine gpuid" 1>&2
  exit 1
fi
name="$1"
host="$2"
gpuid="$3"
desc="Get train per sample loss"
cl work "$(cat cl/cl_worksheet.txt)"
cl run :docqa :nltk_data pred_json:weighted-train/pred.json data:weighted-train/data model:weighted-train/model-0413-081512 'export PYTHONPATH=${PYTHONPATH}:`pwd`;export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 docqa/eval/squad_eval.py -o pred.json --per_question_loss_file per_q.json -c train model*' --request-docker-image robinjia/tf-1.3.0-py3:1.0 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 12g --tail
