#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import time
from collections import Counter
import sys

sys.path.append('..')
import gensim
from torch.autograd import Variable

from utils import *
import torch
import numpy as np
import os
import pickle

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(filename)s %(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

USE_CUDA = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor


def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file, encoding='utf-8') as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


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
# 从embedding file中读取所有的word到一个Set里面
def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""

    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w)
            if valid_words and w not in valid_words:
                continue
            words.add(w)

    if args.restrict_vocab and args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)
        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None

    words = set()
    for ex in examples:
        _insert(ex['question'])
        _insert(ex['document'])
    return words


def build_word_dict(args, examples):
    """Return a dictionary from question and document words in
    provided examples.
    """
    word_dict = Dictionary()
    for w in load_words(args, examples):
        word_dict.add(w)
    return word_dict


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
    logger.info('Load data from file %s ...' % filename)
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(fname=word2vec_file, )
    words2index = {k: v + 1 for v, k in enumerate(word2vec.index2word)}
    data = []
    with open(filename, encoding='utf-8', mode='r') as f:
        wrong_count = 0
        line_count = 0
        wrong = False
        for lines in f:
            lines = [lines.strip(), f.readline().strip(), f.readline().strip()]

            sen = lines[0].strip().split('***')

            data_dict = {}
            w = []
            p = []
            n = []
            line_count += 1
            for s in sen:
                try:
                    w.append(
                        [words2index[w.split(':::')[0]] if words2index.get(w.split(':::')[0]) is not None else 0 for w
                         in s.split('###')])
                    n.append([w.split(':::')[1] for w in s.split(' ')])
                    p.append([w.split(':::')[2] for w in s.split(' ')])
                    wrong = False
                except:
                    '''直接丢掉整个样例'''
                    wrong = True
                    wrong_count += 1
                    print("Get some words wrong, wrong_count %d, line %d" % (wrong_count, line_count))
                    break

            if not wrong:
                data_dict.setdefault('words', w)
                data_dict.setdefault('ners', n)
                data_dict.setdefault('poss', p)
                data_dict.setdefault('fines', [int(lines[1].strip()) - 1])
                data_dict.setdefault('clauses', [int(c) for c in lines[2].strip().split(',')])
                data.append(data_dict)
            # if not USE_CUDA and line_count > 10000:
            if USE_CUDA and line_count > 1000:
                break
        # return words, poss, ners, fines, clauses
        logger.info('Load data from file over. size: %d' % len(data))
        return data, words2index


def load_data(content_only='../data/processed/content_only.txt',
              word2vec_file='../data/processed/word2vec.bin',
              pos_content='../data/processed/pos_content.txt',
              ner_content='../data/processed/ner_content.txt',
              clauses_content='../data/processed/clauses_content.txt',
              fines_content='../data/processed/fines_content.txt', ):
    logger.info('Load data from files ...')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(fname=word2vec_file, )
    words2index = {k: v + 1 for v, k in enumerate(word2vec.index2word)}
    words2index['<Unknow>'] = 0
    embed_arr = word2vec.syn0
    if not os.path.isfile('../data/processed/data.bin'):
        data = []
        content_dic = {}
        pos_dic = {}
        ner_dic = {}
        clauses_dic = {}
        fines_dic = {}

        content_only_file = open(content_only, encoding='utf-8', mode='r')
        pos_file = open(pos_content, encoding='utf-8', mode='r')
        ner_file = open(ner_content, encoding='utf-8', mode='r')
        clauses_file = open(clauses_content, encoding='utf-8', mode='r')
        fines_file = open(fines_content, encoding='utf-8', mode='r')

        for line in content_only_file.readlines():
            idd = line.strip().split('\t')
            sen = idd[1].split(' ')
            idd = int(idd[0])
            content_dic.setdefault(idd, sen)
        for line in pos_file.readlines():
            idd = line.strip().split('\t')
            sen = idd[1].split(' ')
            idd = int(idd[0])
            pos_dic.setdefault(idd, sen)
        for line in ner_file.readlines():
            idd = line.strip().split('\t')
            sen = idd[1].split(' ')
            idd = int(idd[0])
            ner_dic.setdefault(idd, sen)
        for line in clauses_file.readlines():
            idd = line.strip().split('\t')
            sen = idd[1].split(',')
            idd = int(idd[0])
            clauses_dic.setdefault(idd, sen)
        for line in fines_file.readlines():
            idd = line.strip().split('\t')
            sen = idd[1].split(' ')
            idd = int(idd[0])
            fines_dic.setdefault(idd, sen[0])

        for k, sentence in content_dic.items():
            data_dict = {}
            if ner_dic.get(k) is not None and pos_dic.get(k) is not None and clauses_dic.get(
                    k) is not None and fines_dic.get(k) is not None:
                # 拆分文章

                ner = ner_dic.get(k)
                pos = pos_dic.get(k)
                ner_list = []
                pos_list = []
                word_list = []

                ner_sub_list = []
                pos_sub_list = []
                word_sub_list = []

                for i, w in enumerate(sentence):
                    if w == '。' or w == '，' or w == '：' or i == len(sentence) - 1:
                        if len(word_sub_list) > 0:
                            ner_list.append(ner_sub_list.copy())
                            ner_sub_list.clear()
                            pos_list.append(pos_sub_list.copy())
                            pos_sub_list.clear()
                            word_list.append(word_sub_list.copy())
                            word_sub_list.clear()
                    else:
                        word_sub_list.append(words2index[w])
                        ner_sub_list.append(ner[i])
                        pos_sub_list.append(pos[i])

                data_dict.setdefault('words', word_list)
                data_dict.setdefault('ners', ner_list)
                data_dict.setdefault('poss', pos_list)
                data_dict.setdefault('fines', [int(fines_dic.get(k).strip()) - 1])  # pytorch 已经改版，class预测的是从0开始的
                data_dict.setdefault('clauses', [int(c) for c in clauses_dic.get(k)])

                data.append(data_dict)
        content_only_file.close()
        pos_file.close()
        ner_file.close()
        clauses_file.close()
        fines_file.close()

        with open('../data/processed/data.bin', mode='bw+') as f:
            pickle.dump(data, f)
        logger.info('Load the data over , size : %d, save it ../data/processed/data.bin' % len(data))
    else:
        logger.info('Load the data from pickle file...')
        with open('../data/processed/data.bin', 'rb') as f:
            data = pickle.load(f)
    return data, words2index, embed_arr


def collate_batch(batch):
    """
    这里主要是做mask
    [words, features fines, clauses] in one batch list
    Gather a batch of individual examples into one batch.
    here will be batch_size * samples * sentences * seq_len
    pad and pack the sample's length
    :param data:
    :return: a list of batch apply Variable
    """
    words = [d[0] for d in batch]  # batch_size * sentences * seq_len
    words_features = [d[1] for d in batch]  # batch_size * sentences * seq_len * feature_len
    words_mask = [d[2] for d in batch]  # batch_size * sentences * seq_len * feature_len
    fines = [d[3] for d in batch]
    clauses = torch.cat([d[4] for d in batch], 0)

    sort_data = sorted(zip(words, words_features, words_mask, fines, clauses), key=lambda x: len(x[0]), reverse=True)

    words, words_features, words_mask, clauses, fines = zip(*sort_data)

    batch_size = len(batch)  # 这里torch 给的batch 是个list
    lengths = [len(d) for b in words for d in b]  # words count in sentences for each sample

    sentences_count = [len(d) for d in words]

    max_seq_len = max(lengths)
    max_sen_len = max(sentences_count)
    # max_sen_len = 30

    document = torch.LongTensor(batch_size, max_sen_len, max_seq_len).zero_()
    document_mask = torch.ByteTensor(batch_size, max_sen_len, max_seq_len).fill_(1)
    if words_features[0] is not None:
        features = torch.FloatTensor(batch_size, max_sen_len, max_seq_len, words_features[0].size(2)).zero_()
    else:
        features = None
    for i, d in enumerate(words):
        for j in range(max_sen_len):  # 句子
            if j >= len(d):
                break
            s = d[j]
            senten_len = words_mask[i][j].eq(0).long().sum(0)[0]
            document[i, j, :senten_len].copy_(s[:senten_len])
            document_mask[i, j, :senten_len].fill_(0)  # 0 for real
            if features is not None:
                features[i, j, :s.size(0), :].copy_(words_features[i][j])

    document = Variable(document).cuda() if USE_CUDA else Variable(document)
    document_mask = Variable(document_mask).cuda() if USE_CUDA else Variable(document_mask)
    features = Variable(features).cuda() if USE_CUDA else Variable(features)
    fines = Variable(LongTensor(fines)).cuda() if USE_CUDA else Variable(LongTensor(fines))
    # clauses = Variable(clauses).cuda() if USE_CUDA else Variable(clauses)
    return document.long(), document_mask, features, sentences_count, clauses, fines


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

    sens_len = len(data['words'])
    seq_len = [len(d) for d in data['words']]

    # Create extra features vector
    if len(feature_dict) > 0:
        features = torch.zeros(sens_len, max(seq_len), len(feature_dict))  # 注意，直接使用LongTensor，必须要进行初始化，不然默认是有值的，很大且不固定
    else:
        features = None

    document = torch.zeros(sens_len, max(seq_len)).long()  # 单个样例 sen_num * sen_len, 不用段落，直接句子然后文章
    document_mask = torch.ones(sens_len, max(seq_len)).byte()  # 单个样例 sen_num * sen_len, 不用段落，直接句子然后文章
    for i, w in enumerate(data['words']):
        document[i, :len(w)].copy_(torch.from_numpy(np.asarray(w)))
        document_mask[i, :len(w)].fill_(0)  # 0 for pad , 1 for true

    target_tags = torch.zeros(1, len(data['clauses']))
    # for i, t in data['clausea']:
    #     target_tags[i,:len(t)].copy_(LongTensor(t))

    classify = torch.from_numpy(np.asarray(data['fines']))

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
    return document.long(), features, document_mask, target_tags, classify


def build_feature_dict(args, examples):
    """Index features (one hot) from fields in examples and options."""
    logger.info('Build features...')

    def _insert(feature):
        if feature not in feature_dict:
            feature_dict[feature] = len(feature_dict)

    feature_dict = {}

    # Part of speech tag features
    if args.use_pos:
        for ex in examples:
            for w in ex['poss']:
                for p in w:
                    _insert('pos=%s' % p)

    # Named entity tag features
    if args.use_ner:
        for ex in examples:
            for w in ex['ners']:
                for n in w:
                    _insert('ner=%s' % n)

    # Term frequency feature
    if args.use_tf:
        _insert('tf')
    logger.info('Build features over. size : %d' % len(feature_dict))
    print(feature_dict)
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
