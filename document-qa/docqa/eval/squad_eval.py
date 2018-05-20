import argparse
import json
from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from docqa import trainer
from docqa.data_processing.qa_training_data import ParagraphAndQuestionDataset, ContextAndQuestion
from docqa.dataset import FixedOrderBatcher
from docqa.evaluator import Evaluator, Evaluation, SpanEvaluator, ConfidenceSpanEvaluator
from docqa.squad.squad_official_evaluation import exact_match_score as squad_em_score
from docqa.squad.squad_official_evaluation import f1_score as squad_f1_score
from docqa.eval.ranked_squad_scores import compute_model_scores
from docqa.model_dir import ModelDir
from docqa.squad.squad_data import SquadCorpus, split_docs
from docqa.utils import transpose_lists, print_table

"""
Run an evaluation on squad and record the official output
"""


class RecordSpanPrediction(Evaluator):
    def __init__(self, bound: int):
        self.bound = bound

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        return dict(spans=span, model_scores=score)

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
        spans, model_scores = kargs["spans"], kargs["model_scores"]
        results = {"model_conf": model_scores,
                   "predicted_span": spans,
                   "question_id": [x.question_id for x in data]}
        return Evaluation({}, results)

class RecordSpanPredictionScore(Evaluator):
    def __init__(self, bound: int, batch_size: int, include_none_prob: bool):
        self.bound = bound
        self.batch_size = batch_size
        self.include_none_prob = include_none_prob

    def tensors_needed(self, prediction):
        span, score = prediction.get_best_span(self.bound)
        loss = tf.get_collection("PER_SAMPLE_LOSSES")
        loss = tf.add_n(loss, name="loss")
        needed = dict(spans=span, model_scores=score, losses=loss)
        if self.include_none_prob and hasattr(prediction, "none_prob"):
            needed["none_probs"] = prediction.none_prob
        return needed

    def evaluate(self, data: List[ContextAndQuestion], true_len, **kargs):
        spans, model_scores, pred_losses = kargs["spans"], kargs["model_scores"], kargs["losses"]
        print("Len losses: %s, len spans: %s" % (len(pred_losses), len(spans)))
        pred_none_probs = kargs.get("none_probs", None)
        if pred_none_probs is not None:
            print("Len pred_none_probs: %d" % (len(pred_none_probs)))
        pred_f1s = np.zeros(len(data))
        pred_em = np.zeros(len(data))
        for i in tqdm(range(len(data)), total=len(data), ncols=80, desc="scoring"):
            point = data[i]
            f1 = 0.0
            em = 0.0
            if point.answer is not None:
                pred_span = spans[i]
                pred_text = point.paragraph.get_original_text(pred_span[0], pred_span[1])
                f1 = 0
                em = False
                for answer in data[i].answer.answer_text:
                    f1 = max(f1, squad_f1_score(pred_text, answer))
                    if not em:
                        em = squad_em_score(pred_text, answer)

            pred_f1s[i] = f1
            pred_em[i] = em

        results = {"model_confs": model_scores,
                   "predicted_spans": spans,
                   "question_ids": [x.question_id for x in data],
                   "text_f1": pred_f1s,
                   "text_em": pred_em,
                   "loss": pred_losses}
        if pred_none_probs:
            results["none_probs"] = pred_none_probs
        return Evaluation({}, results)


def main():
    parser = argparse.ArgumentParser(description='Evaluate a model on SQuAD')
    parser.add_argument('model', help='model directory to evaluate')
    parser.add_argument("-o", "--official_output", type=str, help="where to output an official result file")
    parser.add_argument('-n', '--sample_questions', type=int, default=None,
                        help="(for testing) run on a subset of questions")
    parser.add_argument('--answer_bounds', nargs='+', type=int, default=[17],
                        help="Max size of answer")
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help="Batch size, larger sizes can be faster but uses more memory")
    parser.add_argument('-s', '--step', default=None,
                        help="Weights to load, can be a checkpoint step or 'latest'")
    parser.add_argument('-c', '--corpus', choices=["dev", "train"], default="dev")
    parser.add_argument('--no_ema', action="store_true", help="Don't use EMA weights even if they exist")
    parser.add_argument('--none_prob', action="store_true", help="Output none probability for samples")
    parser.add_argument('--elmo', action="store_true", help="Use elmo model")
    parser.add_argument('--per_question_loss_file', type=str, default=None,
            help="Run question by question and output a question_id -> loss output to this file")
    args = parser.parse_known_args()[0]

    model_dir = ModelDir(args.model)

    corpus = SquadCorpus()
    if args.corpus == "dev":
        questions = corpus.get_dev()
    else:
        questions = corpus.get_train()
    questions = split_docs(questions)

    if args.sample_questions:
        np.random.RandomState(0).shuffle(sorted(questions, key=lambda x: x.question_id))
        questions = questions[:args.sample_questions]

    questions.sort(key=lambda x:x.n_context_words, reverse=True)
    dataset = ParagraphAndQuestionDataset(questions, FixedOrderBatcher(args.batch_size, True))

    evaluators = [SpanEvaluator(args.answer_bounds, text_eval="squad")]
    if args.official_output is not None:
        evaluators.append(RecordSpanPrediction(args.answer_bounds[0]))
    if args.per_question_loss_file is not None:
        evaluators.append(RecordSpanPredictionScore(args.answer_bounds[0], args.batch_size, args.none_prob))

    if args.step is not None:
        if args.step == "latest":
            checkpoint = model_dir.get_latest_checkpoint()
        else:
            checkpoint = model_dir.get_checkpoint(int(args.step))
    else:
        checkpoint = model_dir.get_best_weights()
        if checkpoint is not None:
            print("Using best weights")
        else:
            print("Using latest checkpoint")
            checkpoint = model_dir.get_latest_checkpoint()

    model = model_dir.get_model()
    if args.elmo:
        model.lm_model.lm_vocab_file = './elmo-params/squad_train_dev_all_unique_tokens.txt'
        model.lm_model.options_file = './elmo-params/options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json'
        model.lm_model.weight_file = './elmo-params/squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5'
        model.lm_model.embed_weights_file = None


    evaluation = trainer.test(model, evaluators, {args.corpus: dataset},
                              corpus.get_resource_loader(), checkpoint, not args.no_ema)[args.corpus]

    # Print the scalar results in a two column table
    scalars = evaluation.scalars
    cols = list(sorted(scalars.keys()))
    table = [cols]
    header = ["Metric", ""]
    table.append([("%s" % scalars[x] if x in scalars else "-") for x in cols])
    print_table([header] + transpose_lists(table))

    # Save the official output
    if args.official_output is not None:
        quid_to_para = {}
        for x in questions:
            quid_to_para[x.question_id] = x.paragraph

        q_id_to_answers = {}
        q_ids = evaluation.per_sample["question_id"]
        spans = evaluation.per_sample["predicted_span"]
        for q_id, (start, end) in zip(q_ids, spans):
            text = quid_to_para[q_id].get_original_text(start, end)
            q_id_to_answers[q_id] = text

        with open(args.official_output, "w") as f:
            json.dump(q_id_to_answers, f)

    if args.per_question_loss_file is not None:
        print("Saving result")
        output_file = args.per_question_loss_file
        ids = evaluation.per_sample["question_ids"]
        f1s = evaluation.per_sample["text_f1"]
        ems = evaluation.per_sample["text_em"]
        losses = evaluation.per_sample["loss"]

        if args.none_prob:
            none_probs = evaluation.per_sample["none_probs"]
            """
            results = {question_id: {'f1': float(f1), 'em': float(em), 'loss': float(loss), 'none_prob': float(none_prob)} for question_id, f1, em, loss, none_prob in zip(ids, f1s, ems, losses, none_probs)}
            """
            results = {question_id: float(none_prob) for question_id, none_prob in zip(ids, none_probs)}
        else:
            results = {question_id: {'f1': float(f1), 'em': float(em), 'loss': float(loss)} for question_id, f1, em, loss in zip(ids, f1s, ems, losses)}


        with open(output_file, 'w') as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()


