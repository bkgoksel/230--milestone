#!/bin/bash set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 name model train_file dev_file host1 gpuid1 host2 gpuid2" 1>&2
  exit 1
fi
name="$1"
model="$2"
train_file="$3"
dev_file="$4"
host1="$5"
gpuid1="$6"
host2="$7"
gpuid2="$8"
desc="Getting no answer probs of ${model} and evaluating P/R on "

cl work "$(cat cl/cl_worksheet.txt)"
cl run :nltk_data :glove :docqa data.json:$train_file model:$model 'export CUDA_VISIBLE_DEVICES='"${gpuid1}"'; bash docqa/scripts/squad-drop-test.sh model data.json' --request-docker-image bkgoksel/docqa:latest -n "${name}"-train -d "${desc}${train_file}" --request-queue host=${host1} --request-memory 12g
cl run :nltk_data :glove :docqa data.json:$dev_file model:$model 'export CUDA_VISIBLE_DEVICES='"${gpuid2}"'; bash docqa/scripts/squad-drop-test.sh model data.json' --request-docker-image bkgoksel/docqa:latest -n "${name}"-dev -d "${desc}${dev_file}" --request-queue host=${host2} --request-memory 12g
