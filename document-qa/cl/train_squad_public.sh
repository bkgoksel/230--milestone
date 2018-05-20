#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName mode [train flags]" 1>&2
  exit 1
fi
name="$1"
mode="$2"
shift
shift
flags="$@"
desc="DocumentQA ${mode}"
if [ -n "${flags}" ]; then
  desc="${desc}, ${flags}"
fi
cl work "$(cat cl_public_worksheet.txt)"
cl run :squad :docqa :glove :nltk_data :eval_squad.py 'export PYTHONPATH=${PYTHONPATH}:`pwd`; python3 docqa/squad/build_squad_dataset.py; python3 docqa/scripts/ablate_squad.py '"${mode} ${flags}"' model; python3 docqa/eval/squad_eval.py -o pred.json -c dev model*; python eval_squad.py squad/dev-v1.1.json pred.json > eval.json'  --request-docker-image robinjia/tf-1.3.0-py3:1.0 -n "${name}" -d "${desc}" --request-queue host=${host}
