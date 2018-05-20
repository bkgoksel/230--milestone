import argparse
import json

def merge_datasets(in_paths, out_path):
    data = []
    for in_path in in_paths:
        with open(in_path, 'r') as f:
            dataset = json.load(f)
            data.extend(dataset['data'])
    merged_data = {'data': data}
    with open(out_path, 'w') as f:
        json.dump(merged_data, f)


def main():
    parser = argparse.ArgumentParser("Merge multiple SQUAD datasets into a single file")
    parser.add_argument('-i', '--in_paths', nargs='+', help="Paths to datasets to merge (1 or more)", required=True)
    parser.add_argument('-o', '--out_path', help="Path to save the merged dataset at", required=True)
    args = parser.parse_args()
    print("Merging: [%s] into %s" % (', '.join(args.in_paths), args.out_path))
    merge_datasets(args.in_paths, args.out_path)

if __name__ == '__main__':
    main()
