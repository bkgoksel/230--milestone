#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 name trainBundle dev.json machine gpuid" 1>&2
  exit 1
fi
name="$1"
train_bundle="$2"
dev_file="$3"
host="$4"
gpuid="$5"
desc="DocumentQA, test ${train_bundle} on ${dev_file}"
cl work "$(cat cl/cl_worksheet.txt)"
cl run train_bundle:"${train_bundle}" dev.json:"${dev_file}" :evaluate-v2.0.py :docqa :glove :nltk_data :eval_squad.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 docqa/run/run_json.py train_bundle/model* dev.json pred.json --na-prob-file na_prob.json --always-answer-file pred_alwaysAnswer.json; python3 evaluate-v2.0.py dev.json pred.json -o eval.json; python3 evaluate-v2.0.py dev.json pred_alwaysAnswer.json -o eval_pr.json -n na_prob.json -p plots'  --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 16g
