import argparse
from collections import defaultdict
import json
import copy

from nltk.tokenize import sent_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    with open(args.infile, 'r') as f:
        data = json.load(f)
        new_docs = []
        for doc in data['data']:
            new_ctx_to_qas = defaultdict(list)
            for p in doc['paragraphs']:
                context_sents = sent_tokenize(p['context'])
                for qa in p['qas']:
                    new_sents = [sent for sent in context_sents]
                    for a in qa['answers']:
                        new_sents = [sent for sent in new_sents if a['text'] not in sent]
                    new_context = ' '.join(new_sents)
                    if not new_context:
                        continue
                    new_qa = {'question': qa['question'],
                            'id': qa['id'] + '_neg',
                            'answers': []}
                    new_ctx_to_qas[new_context].append(new_qa)

            new_docs.append({'title': doc['title'] + '_neg',
                    'paragraphs': [{'qas': qas, 'context': new_context} for new_context, qas in new_ctx_to_qas.items()]})
        data['data'] += new_docs

    with open(args.outfile, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
  main()
