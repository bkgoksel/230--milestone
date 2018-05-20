#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName train.json na_probs.json threshold machine gpuid [train flags]" 1>&2
  exit 1
fi
name="$1"
input_file="$2"
na_probs_file="$3"
threshold="$4"
host="$5"
gpuid="$6"
desc="Drop ${input_file} with FN threshold ${threshold}"

echo $desc

cl work "$(cat cl/cl_worksheet.txt)"

cl add text '% display table drop-table' .
cl run input.json:"${input_file}" na_prob.json:"${na_probs_file}" :docqa 'python3 docqa/run/drop.py input.json na_prob.json --n-hardest 50 --n-easiest 20 --drop-fn-threshold '"${threshold}" --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 16g
cl add text ' ' .
