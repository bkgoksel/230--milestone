#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName model train.json dev.json machine gpuid [train flags]" 1>&2
  exit 1
fi
name="$1"
model="$2"
train_file="$3"
dev_file="$4"
host="$5"
gpuid="$6"
shift
shift
shift
shift
shift
shift
flags="$@"
desc="Predict no answer probs on ${train_file} and ${dev_file}, use ${model}"

echo $desc

cl work "$(cat cl/cl_worksheet.txt)"

cl run train.json:"${train_file}" dev.json:"${dev_file}" :docqa :glove :nltk_data :elmo-params model:"${model}" 'export PYTHONPATH=${PYTHONPATH}:`pwd`; export CUDA_VISIBLE_DEVICES='"${gpuid}"'; python3 docqa/squad/build_squad_dataset.py --train_file train.json --dev_file dev.json; python3 docqa/eval/squad_eval.py model/model* -o pred_train.json -c train --none_prob --per_question_loss_file na_prob_train.json '"${flags}"'; python3 docqa/run/drop.py train.json na_prob_train.json;  python3 docqa/eval/squad_eval.py model/model* -o pred_dev.json -c dev --none_prob --per_question_loss_file na_prob_dev.json '"${flags}"'; python3 docqa/run/drop.py dev.json na_prob_dev.json' --request-docker-image robinjia/tf-1.3.0-py3:1.0.1 -n "${name}" -d "${desc}" --request-queue host=${host} --request-memory 12g

cl add text "% display image /plots/fn_drop_rate.png width=240" .
cl add bundle $name .
cl add text " " .
