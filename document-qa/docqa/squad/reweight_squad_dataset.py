import pdb
import argparse
import json
from collections import defaultdict
from tqdm import tqdm


def compute_weights(metrics_source, weight_metric):
    print("Reading weights data")
    with open(metrics_source, 'r') as f:
        metrics_data = json.load(f)
    print("Normalizing weights data")
    max_val = max(metrics[weight_metric] for metrics in metrics_data.values())
    assert max_val > 0, "Max loss %f <= 0" % max_val
    print("Maximum value for %s: %f. Normalizing" % (weight_metric, max_val))
    weights_data = defaultdict(lambda: 1.0)
    weights_data.update({qid: metrics[weight_metric]/max_val for qid, metrics in metrics_data.items()})
    print("Weights read")
    return weights_data


def reweight(data_source, weights_dict, out_path, drop_threshold=None):
    print("Reading training data from %s" % (data_source))
    with open(data_source, 'r') as f:
        source_data = json.load(f)

    print("Rewriting weights")
    for data_frame in source_data['data']:
        for para in data_frame['paragraphs']:
            for question in para['qas']:
                question['weight'] = weights_dict[question['id']]
            if drop_threshold is not None:
                para['qas'] = [qu for qu in para['qas'] if (len(qu['answers']) > 0 or qu['weight'] > drop_threshold)]

    with open(out_path, 'w') as f:
        json.dump(source_data, f)
    print("Done.")


def main():
    parser = argparse.ArgumentParser("Reweight SQUAD data given some metrics on question ids")
    parser.add_argument("--train_file", \
            help="Original train file to reweight")
    parser.add_argument("--out_file", \
            help="Path to save the reweighted json")
    parser.add_argument("--metrics_file", \
            help="JSON file that contains the question_id -> metrics mappings")
    parser.add_argument("--weight_metric", default="loss", \
            help="Key of metric to use for weighting from the metrics file")
    parser.add_argument("--drop_threshold", type=float, \
            help="If specified, completely drop samples weighted below this threshold")

    args = parser.parse_known_args()[0]
    weight_data = compute_weights(args.metrics_file, args.weight_metric)
    reweight(args.train_file, weight_data, args.out_file, args.drop_threshold)

if __name__ == "__main__":
    main()
