import logging
from collections import Counter

import gensim

from util import *

logger = logging.getLogger(__file__)
import time


# def index_embedding_words(embedding_file):
#     """Put all the words in embedding_file into a set."""
#     words = set()
#     with open(embedding_file, encoding='utf-8') as f:
#         for line in f:
#             w = Dictionary.normalize(line.rstrip().split(' ')[0])
#             words.add(w)
#     return words
#
#
# def load_embedding(filename):
#     word2vec = gensim.models.KeyedVectors.load_word2vec_format(filename)
#     index2words = word2vec.index2word
#     words2index = {w: i for i, w in enumerate(index2words)}
#     embedding = []
#     for w in index2words:
#         embedding.append(word2vec.word_vec(w))
#
#     return word2vec, words2index, index2words
#
#
# # 从embedding file中读取所有的word到一个Set里面
# def load_words(args, examples):
#     """Iterate and index all the words in examples (documents + questions)."""
#
#     def _insert(iterable):
#         for w in iterable:
#             w = Dictionary.normalize(w)
#             if valid_words and w not in valid_words:
#                 continue
#             words.add(w)
#
#     if args.restrict_vocab and args.embedding_file:
#         logger.info('Restricting to words in %s' % args.embedding_file)
#         valid_words = index_embedding_words(args.embedding_file)
#         logger.info('Num words in set = %d' % len(valid_words))
#     else:
#         valid_words = None
#
#     words = set()
#     for ex in examples:
#         _insert(ex['question'])
#         _insert(ex['document'])
#     return words
#
#
# def build_word_dict(args, examples):
#     """Return a dictionary from question and document words in
#     provided examples.
#     """
#     word_dict = Dictionary()
#     for w in load_words(args, examples):
#         word_dict.add(w)
#     return word_dict

def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""

    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')
    return feature_dict


def load_train_data(filename, word2vec_file):
    '''
    load data . words map to index
    filter out the error data and to long data
    :param filename:
    :return: words2index
    '''
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(fname=word2vec_file, )
    words2index = {k: v for v, k in enumerate(word2vec.index2word)}
    data = []
    with open(filename, encoding='utf-8', mode='r') as f:
        # words = []  # split by 。use the sentences
        # poss = []
        # ners = []
        # fines = []
        # clauses = []
        wrong_count = 0
        line_count = 0
        wrong = False
        for lines in f:
            lines = [lines.strip(), f.readline().strip(), f.readline().strip(), f.readline().strip()]

            sen = lines[0].strip().split(' ')

            data_dict = {}
            w = []
            p = []
            n = []
            for s in sen:
                line_count += 1
                try:
                    w.append([words2index[w.split(',')[0]] for w in s.split('。')])
                    n.append([w.split(',')[1] for w in s.split('。')])
                    p.append([w.split(',')[2] for w in s.split('。')])
                    wrong = False
                except:
                    '''直接丢掉整个样例'''
                    wrong = True
                    wrong_count += 1
                    print("Get some words wrong. %s, wrong_count %d, line %d" % (s.split('。'), wrong_count, line_count))
                    break

            if not wrong:
                data_dict.setdefault('words', w)
                data_dict.setdefault('ners', n)
                data_dict.setdefault('poss', p)
                data_dict.setdefault('fines', [int(lines[1].strip())])
                data_dict.setdefault('clauses', [int(c) for c in lines[2].strip().split(',')])
                data.append(data_dict)

        # return words, poss, ners, fines, clauses
        return data, words2index


def collate_batch(batch):
    """
    这里主要是做mask
    [words, poss, ners, fines, clauses] in one list
    Gather a batch of individual examples into one batch.
    here will be batch_size * samples * sentences * seq_len
    pad and pack the sample's length
    :param data:
    :return: a list of batch apply Variable
    """
    words = [d[0] for d in batch]  # batch_size * sentences * seq_len
    features = [d[1] for d in batch]
    fines = [d[2] for d in batch]
    clauses = [d[3] for d in batch]

    batch_size = batch.size(0)
    lengths = [len(d) for b in words for d in b]  # sentences count for each sample

    sentences_count = [len(d) for d in words]

    max_seq_len = max(lengths)
    max_sen_len = max(sentences_count)

    document = torch.LongTensor(batch_size, max_sen_len, max_seq_len).zero_()
    document_mask = torch.ByteTensor(batch_size, max_sen_len, max_seq_len).fill_(1)
    if features[0] is not None:
        features = torch.LongTensor(batch_size, max_sen_len, max_seq_len, features[0].size(1)).zero_()
    else:
        features = None

    for i, d in enumerate(words):
        for j, s in enumerate(d):
            document[i, j, :s.size(0)].copy_(s)
            document_mask[i, j, :s.size(0)].fill_(0)  # 0 for real
            if features is not None:
                features[i, j, :s.size(0)].copy_(features[i])

    return document, document_mask, features, fines, clauses


def vectorize(data, model):
    """
    这里主要是单个样例的特征向量做出来
    vectorize the data of single sample. (sentences * seq_Len)
    :param data:
    :return: document, feature, mask
    """

    # words_dict = model.words_dict
    feature_dict = model.feature_dict
    args = model.args

    sens_len = [d.size(0) for d in data['words']]
    seq_len = [l.size(0) for d in data['words'] for l in d]

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(max(sens_len), max(seq_len), len(feature_dict))
    else:
        features = None

    document = LongTensor(max(sens_len), max(seq_len))  # 单个样例 sen_num * sen_len, 不用段落，直接句子然后文章

    target_tags = LongTensor(1, len(data['clauses']))

    classify = data['fines']

    if args.use_pos:
        for i, d in enumerate(data['poss']):
            for j, w in enumerate(d):
                f = 'pos=%s' % w
                if f in feature_dict:
                    features[i][j][feature_dict[f]] = 1.0

    # f_{token} (NER)
    if args.use_ner:
        for i, d in enumerate(data['ners']):
            for j, w in enumerate(d):
                f = 'ner=%s' % w
                if f in feature_dict:
                    features[i][j][feature_dict[f]] = 1.0

    # f_{token} (TF)
    if args.use_tf:
        counter = Counter([w.lower() for w in data['words']])
        l = len(data['words'])
        for i, d in enumerate(data['poss']):
            for j, w in enumerate(d):
                features[i][j][feature_dict['tf']] = counter[w.lower()] * 1.0 / l

    return document, features, target_tags, classify


def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""

    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['pos']:
                _insert('pos=%s' % w)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ner']:
                _insert('ner=%s' % w)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')
    return feature_dict


# ------------------------------------------------------------------------------
# Utility classes
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer(object):
    """Computes elapsed time."""

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


if __name__ == '__main__':
    data = load_train_data('../data/precessed/pos_ner_content.txt', '../data/precessed/word2vec.bin')
    print(data)
