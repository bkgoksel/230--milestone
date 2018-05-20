#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName split n_per_orig train.json dev.json machine gpuid [flags]" 1>&2
  exit 1
fi
name="$1"
split="$2"
n_per_orig="$3"
train_file="$4"
dev_file="$5"
host="$6"
gpuid="$7"
shift
shift
shift
shift
shift
shift
shift
flags="$@"
desc="Generate tfidf-${split} set with ${n_per_orig} per positive sample from ${train_file}"

echo $desc

cl work "$(cat cl/cl_worksheet.txt)"

cl run :glove :nltk_data :docqa squad_json_data:squad train-v1.1.json:${train_file} dev-v1.1.json:${dev_file} 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"' ; python3 docqa/squad/build_squad_dataset.py --train_file train-v1.1.json --dev_file dev-v1.1.json;python3 docqa/scripts/dump_squad_distant.py '"$split"' -n '"$n_per_orig"' -o tfidf.json '"$flags" --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 16g
