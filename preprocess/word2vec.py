import multiprocessing
import os
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sys.path.append('..')
from Corenlp_Segment import Sengment

filename = '../data/plain/train.txt'
content_only = '../data/precessed/content_only.txt'
model_file = '../data/precessed/word2vec.model'
word2vec_file = '../data/precessed/word2vec.bin'
pos_ner_content = '../data/precessed/pos_ner_content.txt'

jar_path = '~/Downloads/DrQA/data/corenlp/*:/Users/shane/Downloads/stanford-chinese-corenlp-2017-06-09-models.jar'

seg = Sengment(jar_path, annotation=['ner', 'pos'])


flag = 0
if not os.path.isfile(content_only):
    with open(filename, encoding='utf-8') as f:
        with open(content_only, encoding='utf-8', mode='w+') as f2:
            with open(pos_ner_content, encoding='utf-8', mode='w+') as f3:
                for line in f:
                    sen = line.split('\t')[1].strip()
                    tokens = seg.tokenize(sen)
                    words = ' '.join(tokens.words())
                    f2.write(words + '\n')
                    pairs = tokens.make_pair()
                    pairs_str = [','.join(pair) for pair in pairs]
                    f3.write(' '.join(pairs_str) + '\n' + line.split('\t')[2] + '\n' + line.split('\t')[
                        3] + '\n')

                    # pos = ' '.join(tokens.pos())
                    # ner = ' '.join(tokens.entities())
                    # f3.write(words + '\n' + ner + '\n' + pos + '\n' + line.split('\t')[2] + '\n' + line.split('\t')[
                    #     3] + '\n')
                    if flag % 5000 == 0:
                        print('precessed %d' % flag)
                    flag += 1

# tokenize first
model = Word2Vec(LineSentence(content_only), size=128, window=5, min_count=5,
                 workers=multiprocessing.cpu_count())

# trim unneeded model memory = use(much) less RAM
# model.init_sims(replace=True)
model.save(model_file)
model.wv.save_word2vec_format(word2vec_file, binary=False)
