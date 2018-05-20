#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName model train.json machine gpuid [train flags]" 1>&2
  exit 1
fi
name="$1"
model="$2"
input_file="$3"
host="$4"
gpuid="$5"
shift
shift
shift
shift
shift
flags="$@"
desc="Predict no answer probs on ${input_file}, use ${model}"

echo $desc

cl work "$(cat cl/cl_worksheet.txt)"

cl run input.json:"${input_file}" :docqa :glove :nltk_data :elmo-params model:"${model}" 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"' ; python3 docqa/run/run_json.py model/model* input.json pred.json --na-prob-file na_prob.json; python3 docqa/run/drop.py input.json na_prob.json' --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 12g

cl add text "% display image /plots/fn_drop_rate.png width=240" .
cl add bundle $name .
cl add text " " .
