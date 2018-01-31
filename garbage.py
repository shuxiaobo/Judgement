#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by ShaneSue on 30/01/2018

clauses = {}
fines = {}
with open('data/plain/train.txt') as f:
    for l in f:
        sen = l.strip().split('\t')
        clauses.setdefault(sen[0], sen[3])
        fines.setdefault(sen[0], sen[2])

with open('data/processed2/clauses_content.txt', mode='w+') as clauses_file:
    with open('data/processed2/fines_content.txt', mode='w+') as fines_file:
        with open('data/processed2/content_only.txt', mode='w+') as content_only:
            
            clauses_file.write(clauses[sen[0]] + '\t' + sen[3] + '\n')
            fines_file.write(sen[0] + '\t' + sen[2] + '\n')
