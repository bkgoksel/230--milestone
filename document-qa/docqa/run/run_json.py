"""Generate SQuAD (v2.0) predictions file given SQuAD JSON file."""
import argparse
import json
import numpy as np
import os
import re
import sys
import tensorflow as tf
from tqdm import tqdm

from docqa.data_processing.qa_training_data import ParagraphAndQuestion, ParagraphAndQuestionSpec
from docqa.data_processing.text_utils import NltkAndPunctTokenizer, NltkPlusStopWords
from docqa.doc_qa_models import ParagraphQuestionModel
from docqa.elmo.lm_qa_models import ElmoQaModel
from docqa.model_dir import ModelDir
from docqa.squad.build_squad_dataset import parse_squad_data
from docqa.utils import flatten_iterable, CachingResourceLoader, ResourceLoader

from util import *

OPTS = None

DEFAULT_BEAM_SIZE = 2

def parse_args():
    parser = argparse.ArgumentParser('Generate predictions for a SQuAD JSON file.')
    parser.add_argument('model')
    parser.add_argument('input_file', metavar='input.json')
    parser.add_argument('output_file', metavar='pred.json')
    parser.add_argument('--na-prob-file', metavar='na_prob.json')
    parser.add_argument('--always-answer-file', metavar='pred_alwaysAnswer.json')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_input_data(model):
  data = []
  vocab = set()
  tokenizer = NltkAndPunctTokenizer()
  with open(OPTS.input_file) as f:
    json_data = json.load(f)
  for doc in json_data['data']:
    for paragraph in doc['paragraphs']:
      context = tokenizer.tokenize_with_inverse(paragraph['context'])
      if model.preprocessor is not None:
        context = model.preprocessor.encode_text(question, context)
      context = context.get_context()
      vocab.update(context)
      for qa in paragraph['qas']:
        question = tokenizer.tokenize_sentence(qa['question'])
        vocab.update(question)
        ex = [ParagraphAndQuestion(context, question, None, qa['id'])]
        data.append((paragraph['context'], context, ex))
  return data, sorted(list(vocab))

def main():
  print('Starting...')
  model_dir = ModelDir(OPTS.model)
  model = model_dir.get_model()
  if isinstance(model, ParagraphQuestionModel):
      run_paragraph_model(model_dir, model)
  elif isinstance(model, ElmoQaModel):
      run_elmo_model(model_dir, model)
  else:
      raise ValueError("This script is built to work for ParagraphQuestionModel and ElmoQaModel models only")

def run_elmo_model(model_dir, model):
    input_data, vocab = read_input_data(model)

    # Important! This tells the language model not to use the pre-computed word vectors,
    # which are only applicable for the SQuAD dev/train sets.
    # Instead the language model will use its character-level CNN to compute
    # the word vectors dynamically.
    model.lm_model.lm_vocab_file = './elmo-params/squad_train_dev_all_unique_tokens.txt'
    model.lm_model.options_file = './elmo-params/options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json'
    model.lm_model.weight_file = './elmo-params/squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5'
    model.lm_model.embed_weights_file = None

    print('Loading word vectors...')
    model.set_input_spec(ParagraphAndQuestionSpec(batch_size=None), vocab)
    print('Starting Tensorflow session...')
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with sess.as_default():
        prediction = model.get_prediction()
# Take 0-th here because we know we only truncate to one paragraph
        start_logits_tf = prediction.start_logits[0]
        end_logits_tf = prediction.end_logits[0]
        none_logit_tf = prediction.none_logit[0]
    # Now restore the weights, this is a bit fiddly since we need to avoid restoring the
    # bilm weights, and instead load them from the pre-computed data
    all_vars = tf.global_variables() + tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)
    lm_var_names = {x.name for x in all_vars if x.name.startswith("bilm")}
    vars = [x for x in all_vars if x.name not in lm_var_names]
    model_dir.restore_checkpoint(sess, vars)
    sess.run(tf.variables_initializer([x for x in all_vars if x.name in lm_var_names]))


    pred_obj = {}
    na_prob_obj = {}
    pred_always_ans_obj = {}
    for context_raw, context_toks, ex in tqdm(input_data):
        encoded = model.encode(ex, is_train=False)
        start_logits, end_logits, none_logit = sess.run(
                [start_logits_tf, end_logits_tf, none_logit_tf],
                feed_dict=encoded)
        beam, p_na = logits_to_probs(
                context_raw, context_toks, start_logits, end_logits, none_logit,
                beam_size=2)
        ans = beam[0][0]
        non_empty_ans = [x[0] for x in beam if x[0]][0]
        qid = ex[0].question_id
        pred_obj[qid] = ans
        na_prob_obj[qid] = p_na
        pred_always_ans_obj[qid] = non_empty_ans
    with open(OPTS.output_file, 'w') as f:
        json.dump(pred_obj, f)
    if OPTS.na_prob_file:
        with open(OPTS.na_prob_file, 'w') as f:
            json.dump(na_prob_obj, f)
    if OPTS.always_answer_file:
        with open(OPTS.always_answer_file, 'w') as f:
            json.dump(pred_always_ans_obj, f)


def run_paragraph_model(model_dir, model):
  input_data, vocab = read_input_data(model)

  print('Loading word vectors...')
  model.set_input_spec(ParagraphAndQuestionSpec(batch_size=None), vocab)

  print('Starting Tensorflow session...')
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  with sess.as_default():
    prediction = model.get_prediction()
    # Take 0-th here because we know we only truncate to one paragraph
    start_logits_tf = prediction.start_logits[0]
    end_logits_tf = prediction.end_logits[0]
    none_logit_tf = prediction.none_logit[0]
  model_dir.restore_checkpoint(sess)

  pred_obj = {}
  na_prob_obj = {}
  pred_always_ans_obj = {}
  for context_raw, context_toks, ex in tqdm(input_data):
    encoded = model.encode(ex, is_train=False)
    start_logits, end_logits, none_logit = sess.run(
        [start_logits_tf, end_logits_tf, none_logit_tf],
        feed_dict=encoded)
    beam, p_na = logits_to_probs(
        context_raw, context_toks, start_logits, end_logits, none_logit,
        beam_size=2)
    ans = beam[0][0]
    non_empty_ans = [x[0] for x in beam if x[0]][0]
    qid = ex[0].question_id
    pred_obj[qid] = ans
    na_prob_obj[qid] = p_na
    pred_always_ans_obj[qid] = non_empty_ans
  with open(OPTS.output_file, 'w') as f:
    json.dump(pred_obj, f)
  if OPTS.na_prob_file:
    with open(OPTS.na_prob_file, 'w') as f:
      json.dump(na_prob_obj, f)
  if OPTS.always_answer_file:
    with open(OPTS.always_answer_file, 'w') as f:
      json.dump(pred_always_ans_obj, f)

if __name__ == '__main__':
  OPTS = parse_args()
  main()
