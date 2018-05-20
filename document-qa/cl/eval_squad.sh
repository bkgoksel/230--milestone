#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 dev.json pred.json na_prob.json always_answer.json machine gpuid" 1>&2
  exit 1
fi
dev_file="$1"
pred_file="$2"
na_prob_file="$3"
always_answer_file="$4"
host="$5"
gpuid="$6"
desc="Eval on $pred_file $na_prob_file on $dev_file"
cl work "$(cat cl/cl_worksheet.txt)"
cl run pred.json:"${pred_file}" na_prob.json:"${na_prob_file}" pred_alwaysAnswer.json:"${always_answer_file}" dev.json:"${dev_file}" :evaluate-v2.0.py :docqa :glove :nltk_data :eval_squad.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 evaluate-v2.0.py dev.json pred.json -o eval.json; python3 evaluate-v2.0.py dev.json pred_alwaysAnswer.json -o eval_pr.json -n na_prob.json -p plots'  --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "docqa-eval" -d "${desc}" --request-queue host=${host}
