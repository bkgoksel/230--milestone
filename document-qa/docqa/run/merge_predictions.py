import argparse
import json
import sys

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser('Merge predictions of multiple models, input models in order of active training')
    parser.add_argument('--thresholds', '-t', nargs='+')
    parser.add_argument('--na_probs', '-n', nargs='+')
    parser.add_argument('--preds', '-p', nargs='+')
    parser.add_argument('--always_answer_preds', '-a', nargs='+')
    parser.add_argument('--na_prob_file', default='na_prob.json')
    parser.add_argument('--pred_file', default='pred.json')
    parser.add_argument('--always_answer_file', default='pred_alwaysAnswer.json')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def get_file(fl):
    with open(fl) as f:
        return json.load(f)

def main():
    assert len(OPTS.na_probs) == len(OPTS.thresholds)
    assert len(OPTS.na_probs) == len(OPTS.preds)
    print('Filtering no answer predictions')

    final_na_probs = {}
    final_preds = {}
    final_always_answer_preds = {}

    thresholds = [get_file(t) for t in OPTS.thresholds]
    na_probs = [get_file(prob) for prob in OPTS.na_probs]
    preds = [get_file(pred) for pred in OPTS.preds]
    always_answer_preds = [get_file(pred) for pred in OPTS.always_answer_preds]
    thresholds[-1] = 0

    all_qids = [qid for qid, pred in preds[0].items()]
    model_qids = []
    for i, t in enumerate(thresholds):
        new_all_qids = []
        this_model_qids = []
        for qid in all_qids:
            if na_probs[i][qid] > t:
                this_model_qids.append(qid)
            else:
                new_all_qids.append(qid)
        all_qids = new_all_qids
        model_qids.append(this_model_qids)

    for model, qids in enumerate(model_qids):
        for qid in qids:
            final_na_probs[qid] = na_probs[model][qid]
            final_preds[qid] = preds[model][qid]
            final_always_answer_preds[qid] = always_answer_preds[model][qid]


    with open(OPTS.na_prob_file, 'w') as f:
        json.dump(final_na_probs, f)

    with open(OPTS.pred_file, 'w') as f:
        json.dump(final_preds, f)

    with open(OPTS.always_answer_file, 'w') as f:
        json.dump(final_always_answer_preds, f)

if __name__ == '__main__':
    OPTS = parse_args()
    main()
