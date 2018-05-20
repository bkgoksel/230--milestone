#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName mode train.json numBoostingRounds machine gpuid [train flags]" 1>&2
  exit 1
fi
name="$1"
mode="$2"
train_file="$3"
num_boosting_rounds="$4"
host="$5"
gpuid="$6"
shift
shift
shift
shift
shift
shift
flags="$@"
desc="DocumentQA ${mode}, train on ${train_file}, boosting"
if [ -n "${flags}" ]; then
  desc="${desc}, ${flags}"
fi
cl work "$(cat cl/cl_worksheet.txt)"
cl run train-v1.1.json:"${train_file}" :dev-v1.1.json :docqa :glove :nltk_data :eval_squad.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"';bash docqa/scripts/train_squad_boost.sh '"${mode} ${num_boosting_rounds} ${flags}"''  --request-docker-image robinjia/tf-1.3.0-py3:1.0 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 12g
