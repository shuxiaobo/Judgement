#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 23/03/2018
import numpy as np
import pickle
import logging
import gensim
import os
from collections import Iterable

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(filename)s %(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def load_embeddings(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    return word_embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data(content_only='../data/processed/content_only.txt',
              word2vec_file='../data/processed/word2vec.bin',
              pos_content='../data/processed/pos_content.txt',
              ner_content='../data/processed/ner_content.txt',
              clauses_content='../data/processed/clauses_content.txt',
              fines_content='../data/processed/fines_content.txt', seq_len=100, sen_len=14):
    logger.info('Load data from files ...')
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(fname=word2vec_file, )
    words2index = {k: v + 1 for v, k in enumerate(word2vec.index2word)}
    words2index['<Unknow>'] = 0
    embed_arr = word2vec.syn0
    if not os.path.isfile('../data/processed/data_tf.bin'):
        data = []
        y = []
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
                    if w == '。' or w == '，' or w == '：' or i == len(sentence) - 1 or len(word_sub_list) > 14:
                        if len(word_sub_list) > 0:
                            # ner_list.append(ner_sub_list.copy())
                            # ner_sub_list.clear()
                            # pos_list.append(pos_sub_list.copy())
                            # pos_sub_list.clear()
                            word_list.extend(word_sub_list.copy() + [0] * (sen_len - len(word_sub_list)))
                            word_sub_list.clear()
                    else:
                        if w in words2index:
                            word_sub_list.append(words2index[w])
                        else:
                            word_sub_list.append(0)
                        # ner_sub_list.append(ner[i])
                        # pos_sub_list.append(pos[i])
                    if seq_len > 0 and len(word_list) >= seq_len * sen_len:
                        word_list = word_list[:seq_len * sen_len]
                        break
                if len(word_list) < seq_len * sen_len:
                    word_list.extend([0] * (seq_len * sen_len - len(word_list)))

                data_dict.setdefault('words', word_list)
                # data_dict.setdefault('ners', ner_list)
                # data_dict.setdefault('poss', pos_list)
                data_dict.setdefault('fines', [int(fines_dic.get(k).strip()) - 1])  # pytorch 已经改版，class预测的是从0开始的
                # data_dict.setdefault('clauses', [int(c) for c in clauses_dic.get(k)])

                data.extend(word_list)
                y.append([int(fines_dic.get(k).strip()) - 1])
        content_only_file.close()
        pos_file.close()
        ner_file.close()
        clauses_file.close()
        fines_file.close()

        with open('../data/processed/data_tf.bin', mode='bw+') as f:
            pickle.dump({'data': data, 'y': y}, f)
        logger.info('Load the data over , size : %d, save it ../data/processed/data.bin' % len(data))
    else:
        logger.info('Load the data from pickle file...')
        with open('../data/processed/data_tf.bin', 'rb') as f:
            data = pickle.load(f)
            y = data['y']
            data = data['data']
    return np.array(data).reshape(-1, seq_len, sen_len), np.asarray(y), words2index


def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x
