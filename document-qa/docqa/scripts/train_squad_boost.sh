#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 mode num_boosting_rounds [train flags]" 1>&2
  exit 1
fi
mode="$1"
num_boosting_rounds="$2"
shift
shift
flags="$@"

export PYTHONPATH=${PYTHONPATH}:`pwd`
python3 docqa/squad/build_squad_dataset.py --train_file train-v1.1.json --dev_file dev-v1.1.json ${flags}
python3 docqa/scripts/ablate_squad.py ${mode} model ${flags}
python3 docqa/eval/squad_eval.py --per_question_loss_file q_metrics.json -o pred.json -c dev model*
python3 docqa/squad/reweight_squad_dataset.py --train_file train-v1.1.json --out_file train-v1.1.e1.json --metrics_file q_metrics.json ${flags}
rm q_metrics.json
rm pred.json
rm -r data/squad

python3 docqa/squad/build_squad_dataset.py --train_file train-v1.1.e1.json --dev_file dev-v1.1.json ${flags}
python3 docqa/scripts/ablate_squad.py ${mode} model ${flags}
python3 docqa/eval/squad_eval.py --per_question_loss_file q_metrics.json -o pred.json -c dev model*
python eval_squad.py dev-v1.1.json pred.json > eval.json
