#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
import sys

sys.path.append('..')
from preprocess.CoreTokenizer import CoreTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')


class Sengment():
    def __init__(self, dir_to_jar, annotation=None):
        self.dir_to_jar = dir_to_jar
        self.annotation = annotation
        self.segment = CoreTokenizer(classpath=dir_to_jar, annotators=annotation)

    def tokenize(self, text):
        return self.segment.tokenize(text)
