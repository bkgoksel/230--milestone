#!/bin/bash
set -eu -o pipefail
if [ $# -eq 0 ]; then
  echo "Usage: $0 bundleName [elmo|docqa|pretrained-model-bundle-name] train.json dev.json machine gpuid1 gpuid2" 1>&2
  exit 1
fi
name=$1
train_method=$2
train_file=$3
dev_file=$4
host=$5
gpuid1=$6
gpuid2=$7

cl work "$(cat cl/cl_worksheet.txt)"
cl add text "### $name" .
# 1. Train model on set of questions or use pretrained model
case "$train_method" in
  elmo)
    model_name=$name-train-elmo
    bash cl/train_elmo.sh $model_name confidence ${train_file} ${host} ${gpuid1}
    ;;
  docqa)
    model_name=$name-train-docqa
    bash cl/train_squad_kerem.sh $model_name confidence ${train_file} ${host} ${gpuid1}
    ;;
  *)
    model_name=$train_method
esac

# 2. Get na_probs for dev questions
bash cl/pred_no_ans.sh $name-pred-no-ans $model_name ${dev_file} ${host} ${gpuid1}

# 3. Generate new set of questions
bash cl/generate_tfidf_questions.sh $name-generate-tfidf-train train 10 train-v1.1.json dev-v1.1.json ${host} ${gpuid2}
bash cl/generate_tfidf_questions.sh $name-generate-tfidf-dev dev 10 train-v1.1.json dev-v1.1.json ${host} ${gpuid2}

# 4. Get na_probs for new set of questions
bash cl/pred_no_ans.sh $name-new-no-ans-train $model_name $name-generate-tfidf-train/tfidf.json ${host} ${gpuid2}
bash cl/pred_no_ans.sh $name-new-no-ans-dev $model_name $name-generate-tfidf-dev/tfidf.json ${host} ${gpuid1}

# 5. Set threshold
echo "Threshold?"
read threshold

# 6. Filter new set of questions given threshold
bash cl/drop_neg.sh $name-new-drop-train $name-generate-tfidf-train/tfidf.json $name-new-no-ans-train/na_prob.json ${threshold} ${host} ${gpuid2}
bash cl/drop_neg.sh $name-new-drop-dev $name-generate-tfidf-dev/tfidf.json $name-new-no-ans-dev/na_prob.json ${threshold} ${host} ${gpuid1}
