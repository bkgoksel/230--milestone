import argparse
import bisect
import json
import os

def parse_args():
    parser = argparse.ArgumentParser('Filter negative Squad dataset by no-answer probabilities')
    parser.add_argument('data_file', type=str)
    parser.add_argument('na_probs_file', type=str)
    parser.add_argument('--n-hardest', type=int, default=0)
    parser.add_argument('--n-easiest', type=int, default=0)
    parser.add_argument('--sample-q-file', type=str, default='sample.json')
    parser.add_argument('-o', '--out_file', type=str, default='dropped.json')
    parser.add_argument('--drop-fn-threshold', type=float)
    parser.add_argument('--out-image-dir', type=str, default='plots')
    return parser.parse_args()

def make_qid_to_para(dataset):
    qid_to_para = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_para[qa['id']] = p
    return qid_to_para

def get_context_and_question(qid, qid_to_para):
    ctx = qid_to_para[qid]['context']
    question = [q for q in qid_to_para[qid]['qas'] if q['id'] == qid][0]
    return ctx, question

def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
    return qid_to_has_ans

def plot_pr_curve(precisions, recalls, out_image, title):
    plt.step(recalls, precisions, color='b', alpha=0.2, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()

def make_precision_recall_eval(scores, na_probs, num_true_pos, qid_to_has_ans,
        out_image=None, title=None):
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    true_pos = 0.0
    cur_p = 1.0
    cur_r = 0.0
    precisions = [1.0]
    recalls = [0.0]
    avg_prec = 0.0
    for i, qid in enumerate(qid_list):
        if qid_to_has_ans[qid]:
            true_pos += scores[qid]
        cur_p = true_pos / float(i+1)
        cur_r = true_pos / float(num_true_pos)
        if i == len(qid_list) - 1 or na_probs[qid] != na_probs[qid_list[i+1]]:
            # i.e., if we can put a threshold after this point
          avg_prec += cur_p * (cur_r - recalls[-1])
          precisions.append(cur_p)
          recalls.append(cur_r)
    if out_image:
        plot_pr_curve(precisions, recalls, out_image, title)
    return {'ap': 100.0 * avg_prec}

def run_precision_recall_analysis(na_probs, qid_to_has_ans, out_image_dir):
    if out_image_dir and not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    num_true_pos = sum(1 for v in qid_to_has_ans.values() if v)
    if num_true_pos == 0:
        return
    oracle_scores = {k: float(v) for k, v in qid_to_has_ans.items()}
    make_precision_recall_eval(
          oracle_scores, na_probs, num_true_pos, qid_to_has_ans,
          out_image=os.path.join(out_image_dir, 'pr_oracle.png'),
          title='Oracle Precision-Recall curve (binary task of HasAns vs. NoAns)')

def make_fn_drop_rate_plot(data, na_probs, qid_to_has_ans, out_image_dir):
    answerable_sorted_probs = [prob for quid, prob in sorted(na_probs.items(), key=lambda x:x[1], reverse=True) if qid_to_has_ans[quid]]
    unanswerable_sorted_probs = [prob for quid, prob in sorted(na_probs.items(), key=lambda x:x[1]) if not qid_to_has_ans[quid]]
    n_answerable = len(answerable_sorted_probs)
    n_unanswerable = len(unanswerable_sorted_probs)

    fn_rates = []
    drop_rates = []
    for i, cutoff_pt in enumerate(answerable_sorted_probs):
        dropped = n_unanswerable - bisect.bisect_left(unanswerable_sorted_probs, cutoff_pt)
        fn_rates.append(float(i)/n_answerable)
        drop_rates.append(float(dropped)/n_unanswerable)
    plot_drop_fn_curve(fn_rates, drop_rates,
          os.path.join(out_image_dir, 'fn_drop_rate.png'),
          'Pct Unanswerables Dropped vs False Negative Rate')

def plot_drop_fn_curve(fn_rates, drop_rates, out_image, title):
    plt.step(fn_rates, drop_rates, color='b', alpha=0.2, where='post')
    plt.fill_between(fn_rates, drop_rates, step='post', alpha=0.2, color='b')
    plt.xlabel('FN Rate')
    plt.ylabel('Pct Negatives Dropped')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.title(title)
    plt.savefig(out_image)
    plt.clf()


def drop_questions(data, na_probs, qid_to_has_ans, threshold, out_file):
    # Get a list of all answerable question ids, sorted by probability of answer
    answerable_sorted_probs = [prob for quid, prob in sorted(na_probs.items(), key=lambda x:x[1], reverse=True) if qid_to_has_ans[quid]]
    threshold_idx = int(len(answerable_sorted_probs) * threshold)
    cutoff_pt = answerable_sorted_probs[threshold_idx]

    dropped_unanswerable = set()
    kept_unanswerable = set()

    def keep_question(qa):
        has_answer = bool(len(qa['answers']))
        below_threshold = na_probs[qa['id']] < cutoff_pt
        keep = has_answer or below_threshold
        if keep and not has_answer:
            kept_unanswerable.add(qa['id'])
        if not keep and not has_answer:
            dropped_unanswerable.add(qa['id'])
        return keep

    for doc in data:
        for paragraph in doc['paragraphs']:
            paragraph['qas'] = [qa for qa in paragraph['qas'] if keep_question(qa)]

    n_kept = len(kept_unanswerable)
    n_dropped = len(dropped_unanswerable)
    n_total = n_kept + n_dropped

    print("%d unanswerable questions: %d kept, %d dropped: (%f dropped)" % ( \
            n_total,
            n_kept,
            n_dropped,
            float(n_dropped)/n_total))

    final_data = {'data': data}

    with open(out_file, 'w') as f:
        json.dump(final_data, f)

    with open('cutoff-%s' % out_file, 'w') as f:
        json.dump(cutoff_pt, f)


def sample_questions(n_hardest, n_easiest, out_file, data, na_probs, qid_to_has_ans):
    qid_to_para = make_qid_to_para(data)
    unanswerable_probs = {k: v for k,v in na_probs.items() if not qid_to_has_ans[k]}
    unanswerable_sorted = sorted(unanswerable_probs, key=lambda k: na_probs[k])
    hardest = []
    for i in range(n_hardest):
        qid = unanswerable_sorted[i]
        # __import__('pdb').set_trace()
        positive_qid = qid[:-4]
        ctx, question = get_context_and_question(qid, qid_to_para)
        positive_ctx, positive_q = get_context_and_question(positive_qid, qid_to_para)
        positive_answer = positive_q['answers']
        na_prob = na_probs[qid]
        hardest.append({'q': question, 'ctx': ctx, 'positive_ctx': positive_ctx, 'positive_answers': positive_answer, 'na_prob': na_prob})
    easiest = []
    for i in range(1, n_easiest+1):
        qid = unanswerable_sorted[-i]
        positive_qid = qid[:-4]
        ctx, question = get_context_and_question(qid, qid_to_para)
        positive_ctx, positive_q = get_context_and_question(positive_qid, qid_to_para)
        positive_answer = positive_q['answers']
        na_prob = na_probs[qid]
        easiest.append({'q': question, 'ctx': ctx, 'positive_ctx': positive_ctx, 'positive_answers': positive_answer, 'na_prob': na_prob})
    out = {'hardest': hardest, 'easiest': easiest}
    with open(out_file, 'w') as f:
        json.dump(out, f)


def main(args):
    with open(args.data_file) as f:
        data = json.load(f)['data']
    with open(args.na_probs_file) as f:
        na_probs = json.load(f)

    qid_to_has_ans = make_qid_to_has_ans(data)
    run_precision_recall_analysis(na_probs, qid_to_has_ans, args.out_image_dir)
    make_fn_drop_rate_plot(data, na_probs, qid_to_has_ans, args.out_image_dir)
    if args.n_hardest or args.n_easiest:
        sample_questions(args.n_hardest, args.n_easiest, args.sample_q_file, data, na_probs, qid_to_has_ans)
    if args.drop_fn_threshold is not None:
        drop_questions(data, na_probs, qid_to_has_ans, args.drop_fn_threshold, args.out_file)

if __name__ == '__main__':
    args = parse_args()
    if args.out_image_dir:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    main(args)
