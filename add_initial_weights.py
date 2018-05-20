import json
import argparse

def main():
    parser = argparse.ArgumentParser("Preprocess SQuAD data")
    parser.add_argument("--in_file", default='train-v1.1.json')
    parser.add_argument("--out_file", default='train-v1.1-weighted.json')
    args = parser.parse_args()

    with open(args.in_file, 'r') as infile:
        data = json.load(infile)
    print("Finished reading in %s" % args.in_file)

    iter_data = data['data']
    for sample in iter_data:
        for parag in sample['paragraphs']:
            for qa in parag['qas']:
                qa['weight'] = 1.0

    print("Finished adding weights")

    with open(args.out_file, 'w') as outfile:
        json.dump(data, outfile)
    print("Finished writing to %s" % args.out_file)

if __name__ == "__main__":
    main()
