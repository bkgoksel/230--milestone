from typing import List, Optional, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

from docqa.data_processing.multi_paragraph_qa import (
        ParagraphWithAnswers,
        MultiParagraphQuestion,
        WeightedMultiParagraphQuestion,
        TokenSpanGroup
)
from docqa.data_processing.preprocessed_corpus import Preprocessor
from docqa.data_processing.qa_training_data import ContextAndQuestion, Answer
from docqa.data_processing.span_data import TokenSpans
from docqa.data_processing.text_utils import WordNormalizer
from docqa.squad.squad_data import Document, SquadCorpus
from docqa.text_preprocessor import TextPreprocessor
from docqa.utils import flatten_iterable

"""
Preprocessors for document-level question answering with SQuAD data
"""


class SquadParagraphWithAnswers(ParagraphWithAnswers):

    @classmethod
    def merge(cls, paras: List):
        paras.sort(key=lambda x: x.get_order())
        answer_spans = []
        text = []
        original_text = ""
        spans = []
        for para in paras:
            answer_spans.append(len(text) + para.answer_spans)
            spans.append(len(original_text) + para.spans)
            original_text += para.original_text
            text += para.text

        para = SquadParagraphWithAnswers(text, np.concatenate(answer_spans),
                                         paras[0].doc_id, paras[0].paragraph_num,
                                         original_text, np.concatenate(spans))
        return para

    __slots__ = ["doc_id", "original_text", "paragraph_num", "spans"]

    def __init__(self, text: List[str], answer_spans: np.ndarray, doc_id: str, paragraph_num: int,
                 original_text: str, spans: np.ndarray):
        super().__init__(text, answer_spans)
        self.doc_id = doc_id
        self.original_text = original_text
        self.paragraph_num = paragraph_num
        self.spans = spans

    def get_order(self):
        return self.paragraph_num

    def get_original_text(self, start, end):
        return self.original_text[self.spans[start][0]:self.spans[end][1]]

    def build_qa_pair(self, question, question_id, answer_text, group=None):
        #if not answer_text:
        if answer_text is None:
            ans = None
        elif group is None:
            ans = TokenSpans(answer_text, self.answer_spans)
        else:
            ans = TokenSpanGroup(answer_text, self.answer_spans, group)
        # returns a context-and-question equiped with a get_original_text method
        return QuestionAndSquadParagraph(question, ans, question_id, self)


class QuestionAndSquadParagraph(ContextAndQuestion):
    def __init__(self, question: List[str], answer: Optional[Answer], question_id: str, para: SquadParagraphWithAnswers):
        super().__init__(question, answer, question_id, para.doc_id)
        self.para = para

    def get_original_text(self, start, end):
        return self.para.get_original_text(start, end)

    def get_context(self):
        return self.para.text

    @property
    def n_context_words(self) -> int:
        return len(self.para.text)


class SquadTfIdfRanker(Preprocessor):
    def __init__(self, stop, n_to_select: int, force_answer: bool, text_process: TextPreprocessor=None, word_normalizer: WordNormalizer=None):
        self.stop = stop
        self.n_to_select = n_to_select
        self.force_answer = force_answer
        self.text_process = text_process
        self.word_normalizer = word_normalizer
        self._tfidf = TfidfVectorizer(strip_accents="unicode", stop_words=self.stop.words)

    def preprocess(self, question: List[Document], evidence):
        return self.ranked_questions(question)

    def get_q_and_para_texts(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        para_texts = [" ".join(" ".join(s) for s in x) for x in paragraphs]
        q_texts = [" ".join(q) for q in questions]

        if self.word_normalizer:
            new_paras = []
            for p in para_texts:
                new_paras.append(" ".join([self.word_normalizer.normalize(w) for w in p.split(" ")]))
            para_texts = new_paras
            new_qs = []
            for q in q_texts:
                new_qs.append(" ".join([self.word_normalizer.normalize(w) for w in q.split(" ")]))
            q_texts = new_qs

        return q_texts, para_texts

    def rank(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        q_texts, para_texts = self.get_q_and_para_texts(questions, paragraphs)
        tfidf = self._tfidf
        para_features = tfidf.fit_transform(para_texts)
        q_features = tfidf.transform(q_texts)
        scores = pairwise_distances(q_features, para_features, "cosine")
        return scores

    def ranked_questions(self, docs: List[Document]) -> List[MultiParagraphQuestion]:
        out = []
        for doc in docs:
            scores = self.rank(flatten_iterable([q.words for q in x.questions] for x in doc.paragraphs),
                               [x.text for x in doc.paragraphs])
            q_ix = 0
            for para_ix, para in enumerate(doc.paragraphs):
                for q in para.questions:
                    para_scores = scores[q_ix]
                    para_ranks = np.argsort(para_scores)
                    selection = [i for i in para_ranks[:self.n_to_select]]

                    if self.force_answer and para_ix not in selection:
                        selection[-1] = para_ix

                    para = []
                    for ix in selection:
                        #if ix == para_ix:
                        if ix == para_ix and q.answer:
                            ans = q.answer.answer_spans
                        else:
                            ans = np.zeros((0, 2), dtype=np.int32)
                        p = doc.paragraphs[ix]
                        if self.text_process:
                            text, ans, inv = self.text_process.encode_paragraph(q.words,  [flatten_iterable(p.text)],
                                                               p.paragraph_num == 0, ans, p.spans)
                            para.append(SquadParagraphWithAnswers(text, ans, doc.doc_id,
                                                                  ix, p.original_text, inv))
                        else:
                            para.append(SquadParagraphWithAnswers(flatten_iterable(p.text), ans, doc.doc_id,
                                                                  ix, p.original_text, p.spans))

                    out.append(MultiParagraphQuestion(q.question_id, q.words, q.answer.answer_text, para))
                    q_ix += 1
        return out


class SquadVectorTfIdfRanker(SquadTfIdfRanker):
    def __init__(self, stop, n_to_select: int, force_answer: bool, vectors: Dict[str, Any], text_process: TextPreprocessor=None, word_normalizer: WordNormalizer=None):
        super().__init__(stop, n_to_select, force_answer, text_process, word_normalizer)
        self._vectors = vectors

    def _vectorize(self, text):
        return np.mean(np.stack([self._vectors[w.lower()] for w in text if (w.lower() not in self.stop.words and w.lower() in self._vectors)]), axis=0)

    def rank(self, questions: List[List[str]], paragraphs: List[List[List[str]]]):
        q_texts, para_texts = self.get_q_and_para_texts(questions, paragraphs)

        para_vecs = np.stack([self._vectorize(p) for p in para_texts])
        q_vecs = np.stack([self._vectorize(q) for q in q_texts])
        # Dimensionality issue RIGHT FUCKING HERE
        vec_scores = pairwise_distances(q_vecs, para_vecs, "cosine")

        tfidf_scores = super().rank(questions, paragraphs)

        return vec_scores * tfidf_scores


class SquadDefault(Preprocessor):
    """Just uses the provided data."""
    def __init__(self, text_process: TextPreprocessor=None):
        self.text_process = text_process

    def preprocess(self, docs: List[Document], evidence, name=None):
        out = []
        for doc in docs:
            for para_ix, para in enumerate(doc.paragraphs):
                for q in para.questions:
                    if q.answer:
                        ans = q.answer.answer_spans
                    else:
                        ans = np.zeros((0, 2), dtype=np.int32)
                    if self.text_process:
                        text, ans, inv = self.text_process.encode_paragraph(
                            q.words,  [flatten_iterable(para.text)],
                            para.paragraph_num == 0, ans, para.spans)
                        new_para = SquadParagraphWithAnswers(
                            text, ans, doc.doc_id, para_ix, para.original_text, inv)
                    else:
                        new_para = SquadParagraphWithAnswers(
                            flatten_iterable(para.text), ans, doc.doc_id,
                            para_ix, para.original_text, para.spans)
                    out.append(MultiParagraphQuestion(q.question_id, q.words, q.answer.answer_text, [new_para]))
        return out

class SquadWeighted(Preprocessor):
    """Just uses the provided data, handles weighted samples."""
    def __init__(self, text_process: TextPreprocessor=None):
        self.text_process = text_process

    def preprocess(self, docs: List[Document], evidence, name='train'):
        out = []
        for doc in docs:
            for para_ix, para in enumerate(doc.paragraphs):
                for q in para.questions:
                    if q.answer:
                        ans = q.answer.answer_spans
                    else:
                        ans = np.zeros((0, 2), dtype=np.int32)
                    if self.text_process:
                        text, ans, inv = self.text_process.encode_paragraph(
                            q.words,  [flatten_iterable(para.text)],
                            para.paragraph_num == 0, ans, para.spans)
                        new_para = SquadParagraphWithAnswers(
                            text, ans, doc.doc_id, para_ix, para.original_text, inv)
                    else:
                        new_para = SquadParagraphWithAnswers(
                            flatten_iterable(para.text), ans, doc.doc_id,
                            para_ix, para.original_text, para.spans)
                    if name == "train":
                        out.append(WeightedMultiParagraphQuestion(q.question_id, q.words, q.answer.answer_text, [new_para], q.weight))
                    else:
                        out.append(MultiParagraphQuestion(q.question_id, q.words, q.answer.answer_text, [new_para]))
        return out
