'''
1. 将结构化文本转换成纯文本，分词训练word2vec
2. 分拆每个案件的事实
3. 分词、每个词的POS、NER
4. char embedding？
5.
'''
import logging

import pandas as pd

logger = logging.getLogger(__file__)


def read_df_from_file(filename):
    data = pd.read_table(filename, header=None, names=['id', 'context', 'fine', 'clauses'])
    return data




