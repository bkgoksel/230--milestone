#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 referenceNaProbs machine gpuid [other na-probs]" 1>&2
  exit 1
fi
host="$1"
gpuid="$2"
shift
shift
args="$@"
desc="Merge predictions of $args"

args_spec=$(echo "$args" | sed 's/\</:/g')
args_cutoff=$(echo "$args" | sed 's/\>/\/cutoff-dropped.json/g')
args_no_prob=$(echo "$args" | sed 's/\>/\/na_prob.json/g')
args_pred=$(echo "$args" | sed 's/\>/\/pred.json/g')
args_always_ans=$(echo "$args" | sed 's/\>/\/pred_alwaysAnswer.json/g')

cl work "$(cat cl/cl_worksheet.txt)"

cl run :docqa $args_spec 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 docqa/run/merge_predictions.py -t '"${args_cutoff}"' -n '"${args_no_prob}"' -p '"${args_pred}"' -a '"${args_always_ans}" --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "merge-preds" -d "${desc}" --request-queue host=${host}
