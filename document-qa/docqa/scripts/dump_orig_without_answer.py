import argparse
import json

from nltk.tokenize import sent_tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    with open(args.infile, 'r') as f:
        new_data = {'data': []}
        orig_data = json.load(f)
        for doc in orig_data['data']:
            new_doc = doc
            new_doc['paragraphs'] = []
            for p in doc['paragraphs']:
                orig_context = p['context']
                context_sents = sent_tokenize(orig_context)
                for qa in p['qas']:
                    new_sents = [sent for sent in context_sents]
                    for a in qa['answers']:
                        new_sents = [sent for sent in new_sents if a['text'] not in sent]
                    new_context = ' '.join(new_sents)
                    new_qa = {'question': qa['question'],
                            'id': qa['id'] + '_neg',
                            'answers': []}
                    new_p = {'qas': [new_qa],
                            'context': new_context}
                    new_doc['paragraphs'].append(new_p)
            new_data['data'].append(new_doc)

    with open(args.outfile, 'w') as f:
        json.dumps(new_data, f)

if __name__ == '__main__':
  main()
