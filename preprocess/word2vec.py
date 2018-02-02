#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import multiprocessing
import os
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sys.path.append('..')
from preprocess.Corenlp_Segment import Sengment

filename = '../data/plain/train.txt'
content_only = '../data/processed/content_only.txt'
model_file = '../data/processed/word2vec.model'
word2vec_file = '../data/processed/word2vec.bin'
pos_content = '../data/processed/pos_content.txt'
ner_content = '../data/processed/ner_content.txt'
clauses_content = '../data/processed/clauses_content.txt'
fines_content = '../data/processed/fines_content.txt'

pos_ner_content = '../data/processed/pos_ner_content.txt'

# jar_path = '~/Downloads/DrQA/data/corenlp/*:/Users/shane/Downloads/stanford-chinese-corenlp-2017-06-09-models.jar'
jar_path = '/home/shuxiaobo/pycharm/data/corenlp/*'

seg = Sengment(jar_path, annotation=['ner', 'pos'])

from functools import partial
from multiprocessing import Manager
from multiprocessing.pool import Pool as workers_pool
from multiprocessing.util import Finalize

mana = Manager()
words_dict = mana.dict()
lock = mana.Lock()


def init():
    global seg
    seg = Sengment(jar_path, annotation=['ner', 'pos'])
    Finalize(seg, seg.segment.shutdown, exitpriority=100)


def tokenize(text):
    global seg
    return seg.tokenize(text)


# flag = 0
# if not os.path.isfile(content_only):
#     with open(filename, encoding='utf-8') as f:
#         with open(content_only, encoding='utf-8', mode='w+') as f2:
#             with open(pos_content, encoding='utf-8', mode='w+') as f3:
#                 with open(ner_content, encoding='utf-8', mode='w+') as f4:
#                     for line in f:
#                         sen = line.split('\t')[1].strip()
#                         if len(sen) > 5 and len(sen) < 3000:
#                             tokens = seg.tokenize(sen)
#                             words = ' '.join(tokens.words())
#                             f2.write(words + '\n')
#                             f3.write(' '.join(tokens.pos()))
#                             f4.write(' '.join(tokens.entities()))
#                             # pairs = tokens.make_pair()
#                             # # pairs_str = [','.join(pair) for pair in pairs]
#                             # f3.write(pairs + '\n' + line.split('\t')[2] + '\n' + line.split('\t')[
#                             #     3] + '\n')
#                             if flag % 5000 == 0:
#                                 print('precessed %d' % flag)
#                             flag += 1

train_file = open(filename, encoding='utf-8')
content_only_file = open(content_only, encoding='utf-8', mode='w+')
pos_file = open(pos_content, encoding='utf-8', mode='w+')
ner_file = open(ner_content, encoding='utf-8', mode='w+')
clauses_file = open(clauses_content, encoding='utf-8', mode='w+')
fines_file = open(fines_content, encoding='utf-8', mode='w+')

pos_ner_file = open(pos_ner_content, encoding='utf-8', mode='w+')


def assign():
    workers = workers_pool(4, initializer=init)
    # seg_fun = partial(pair_segment)
    seg_fun = partial(segment)
    step = 20

    lines = train_file.readlines()
    batches = [lines[i: i + step] for i in range(0, len(lines), step)]
    for i, b in enumerate(batches):
        workers.imap_unordered(seg_fun, b)
        if i % 50 == 0:
            print('precessed %d batchs/ %d' % (i, step))

    workers.close()
    workers.join()

    train_file.close()
    # pos_ner_file.close()
    content_only_file.close()
    pos_file.close()
    ner_file.close()
    clauses_file.close()
    fines_file.close()
    return


def segment(line):
    sen = line.strip().split('\t')
    if len(sen[1]) > 5 and len(sen[1]) < 3000:
        tokens = tokenize(sen[1])
        words = ' '.join(tokens.words())
        # lock.acquire()
        content_only_file.write(sen[0] + '\t' + words + '\n')
        pos_file.write(sen[0] + '\t' + ' '.join(tokens.pos()) + '\n')
        ner_file.write(sen[0] + '\t' + ' '.join(tokens.entities()) + '\n')
        clauses_file.write(sen[0] + '\t' + sen[3] + '\n')
        fines_file.write(sen[0] + '\t' + sen[2] + '\n')
        # lock.release()


def pair_segment(line):
    sen = line.strip().split('\t')
    if len(sen[1]) > 5 and len(sen[1]) < 3000:
        tokens = tokenize(sen[1])
        # words = ' '.join(tokens.words())
        pairs = tokens.make_pair()
        pos_ner_file.write(pairs + '\n' + sen[2] + '\n' + sen[3] + '\n')
        # content_only_file.write(words + '\n')


if __name__ == '__main__':
    assign()

    # tokenize first
    model = Word2Vec(LineSentence(content_only), size=128, window=5, min_count=0,
                     workers=multiprocessing.cpu_count())
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model.save(model_file)
    model.wv.save_word2vec_format(word2vec_file, binary=False)
