#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json

import pexpect
import sys

sys.path.append('..')
from preprocess.Tokenizer import Tokens, Tokenizer


class CoreTokenizer(Tokenizer):
    def __init__(self, **kargs):
        '''
        classpath always :
        java -mx2G -cp ~/Downloads/DrQA/data/corenlp/*:/Users/shane/Downloads/stanford-chinese-corenlp-2017-06-09-models.jar edu.stanford.nlp.pipeline.StanfordCoreNLP -props StanfordCoreNLP-chinese.properties -annotators tokenize,ssplit -tokenize.options untokenizable=noneDelete,invertible=true -outputFormat json -prettyPrint false
        :param kargs:
        '''
        self.classpath = kargs.get('classpath')

        if not self.classpath:
            raise ValueError("No classpath explict!")

        self.annotators = kargs.get('annotators', set())

        self.memory = kargs.get('memory', '2G')
        self._lanuch()

    def _lanuch(self):
        annotators = ['tokenize', 'ssplit']

        # here annotators has some spacial relation
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete',
                            'invertible=true'])
        cmd = ['java', '-mx' + self.memory, '-cp', '%s' % self.classpath,
               'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-props', 'StanfordCoreNLP-chinese.properties',
               '-annotators',
               annotators, '-tokenize.options', options,
               '-outputFormat', 'json', '-prettyPrint', 'false']

        self.corenlp = pexpect.spawn('/bin/bash', maxread=1000000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')  # stty是用来设置终端的，禁用规范输入。
        self.corenlp.sendline(' '.join(cmd))  # join 是把数组变成字符串，每个元素之间用空格隔开
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    @staticmethod
    def _convert(token):
        if token == '-LRB-':
            return '('
        elif token == '-RRB-':
            return ')'
        elif token == '-LSB-':
            return '['
        elif token == '-RSB-':
            return ']'
        elif token == '-LCB-':
            return '{'
        elif token == '-RCB-':
            return '}'
        return token

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing
        # the NLP> prompt. Hacky!
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text and (text.strip() == 'q' or text.strip() == 'Q'):
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return Tokens(data, self.annotators)

        # Minor cleanup before tokenizing.
        clean_text = text.replace('\n', ' ')

        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=10000)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))

        data = []
        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i]['characterOffsetBegin']
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1]['characterOffsetBegin']
            else:
                end_ws = tokens[i]['characterOffsetEnd']

            data.append((
                self._convert(tokens[i]['word']),
                text[start_ws: end_ws],
                (tokens[i]['characterOffsetBegin'],
                 tokens[i]['characterOffsetEnd']),
                tokens[i].get('pos', None),
                tokens[i].get('lemma', None),
                tokens[i].get('ner', None)
            ))
        return Tokens(data, self.annotators)


if __name__ == '__main__':
    tokenizer = CoreTokenizer()
    token = tokenizer.tokenize()
    print(token.ngrams())
