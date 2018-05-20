#!/bin/bash set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 model data_file" 1>&2
  exit 1
fi
model="$1"
data_file="$2"

export PYTHONPATH=${PYTHONPATH}:`pwd`
echo 'Getting negative probabilities for file '"$data_file"
python3 docqa/run/run_json.py $model $data_file pred.json --na-prob-file na_probs.json
echo 'P/R negative probability evaluating file'
python3 docqa/run/drop.py $data_file na_probs.json
