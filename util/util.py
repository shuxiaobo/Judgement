import gensim


def load_embedding(filename):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(filename)
    index2words = word2vec.index2word
    words2index = {w: i for i, w in enumerate(index2words)}
    embedding = []
    for w in index2words:
        embedding.append(word2vec.word_vec(w))

    return word2vec, words2index, index2words


def load_train_data(filename, words2index):
    '''
    load data . words map to index
    :param filename:
    :return:
    '''
    data = []
    with open(filename, encoding='utf-8', mode='r') as f:
        words = []  # split by 。use the sentences
        poss = []
        ners = []
        fines = []
        clauses = []
        for lines in f.readlines(hint=3):
            sen = []
            for all in lines[0].strip().split(' '): #
                all = all.split(',')
                sen.append()

            sen = [lines[0].strip().split(' ')]
            # ner = [lines[1].strip().split(' ')]
            # pos = [lines[2].strip().split(' ')]
            # fin = [int(lines[3].strip())]
            # clause = [int(c) for c in lines[4].strip().split(',')]
            for s in sen:
                s = [words2index(w) for w in s]

            words.append([lines[0].strip().split('。')])
            ners.append([ners])
            ners.append([lines[1].strip().split(' ')])
            poss.append([lines[1].strip().split(' ')])
            fines.append([int(lines[3].strip())])
            clauses.append([int(c) for c in lines[4].strip().split(',')])

        return words, poss, ners, fines, clauses
