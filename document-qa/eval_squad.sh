#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 name predsBundle dev.json machine" 1>&2
  exit 1
fi
name="$1"
dev_preds="$2"
dev_file="$3"
host="$4"
desc="Evaluate on squadrun"

cl work "$(cat cl/cl_worksheet.txt)"
cl run dev_preds:"${dev_preds}" dev.json:"${dev_file}" :evaluate-v2.0.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; python3 evaluate-v2.0.py dev.json pred.json -o eval.json; python3 evaluate-v2.0.py dev.json pred_alwaysAnswer.json -o eval_pr.json -n na_prob.json -p plots'  --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "${name}" -d "${desc}" --request-queue host=${host}
